import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Union

# noinspection PyUnresolvedReferences
import stream_ep_cpp
# noinspection PyUnresolvedReferences
from stream_ep_cpp import Config, EventHandle
from .utils import check_nvlink_connections


@dataclass
class StreamingHandle:
    """All per-dispatch state for the streaming-MoE pipeline (pool layout).
    Returned by ``Buffer.dispatch``; consumed by the kernel A / Y / combine-gate
    stages and by ``Buffer.combine``.

    Pool layout: dispatch's receiver writes each landed (token, k) pair routing
    to a local expert into its own pool slot. Pool is laid out expert-major,
    BLOCK_M-padded, with per-expert blocks at ``expert_pool_block_offset[e] *
    tile_m`` rows.
    """

    # ── Pool data (kernel A reads with strided TMA; kernel Y / combine read via
    # pool_recv_token + pool_k_slot scatter-back).
    pool: torch.Tensor                         # [TK_padded, hidden] data
    pool_topk_weight: torch.Tensor             # [TK_padded] float32, per-pool-slot weight
    pool_recv_token: torch.Tensor              # [TK_padded] int32, slot → recv-token id (-1 = padding)
    pool_k_slot: torch.Tensor                  # [TK_padded] int32, slot → k (-1 = padding)

    # ── Combine inputs (per-recv-token; combine sender forwards these back to source).
    send_head: torch.Tensor                    # [num_tokens, num_ranks] int32
    is_token_in_rank: torch.Tensor             # [num_tokens, num_ranks] bool
    rank_prefix_matrix: torch.Tensor           # [num_ranks, num_ranks] int32 (this rank's column populated)
    channel_prefix_matrix: torch.Tensor        # [num_ranks, num_channels] int32 (sender-side)
    recv_channel_prefix_matrix: torch.Tensor   # [num_ranks, num_channels] int32 (receiver-side)

    # ── Pool metadata (kernel A scheduler + tests).
    expert_frequency: torch.Tensor             # [E_local] int32
    expert_pool_block_offset: torch.Tensor     # [E_local + 1] int32 — pool-block prefix-sum (BLOCK_M-tile units)
    base_pool: torch.Tensor                    # [num_channels, num_ranks, E_local] int32
    seen_per_substream: torch.Tensor           # [num_channels, num_ranks, E_local] int32 — bwd Pass 2 input (substream's per-expert recv count)

    # ── Per-tile arrays.
    tile_id_to_expert: torch.Tensor            # [total_tiles] int32 — per-tile expert lookup
    pool_arrival_target: torch.Tensor          # [total_tiles] int32 (per-tile firing-target)
    # Per-tile ready signal: dispatch's Pass 2 release-stores `dispatch_seq` into
    # `tile_ready[tile_id]` once pool_arrival_count[tile_id] reaches its target.
    # Pass 2 walks experts in order, so tile fires arrive in expert-monotonic
    # order across substream blocks — preserves wave caching of W1[e].
    tile_ready: torch.Tensor                   # [total_tiles] int64

    # ── Kernel A → kernel Y / kernel Y → combine pipeline buffers.
    # All allocated on dispatch_stream; cross-stream visibility carried by the
    # per-tile release/acquire pairs `tile_ready` (dispatch→A), `a_ready` (A→Y),
    # and `compute_done_per_token` (Y→combine).
    a_ready: torch.Tensor                      # [total_tiles] int64 — A→Y per-tile release stamp (zero-init)
    per_token_remaining: torch.Tensor          # [T_recv] int32 — K_local(r); kernel Y atomicSubs
    compute_done_per_token: torch.Tensor       # [T_recv] int64 — Y→combine per-token release stamp (zero-init)
    o: torch.Tensor                            # [T_recv, hidden] — kernel Y atomic-scatter destination (zero-init)

    # ── Backward-pass scaffolding (Phase F).
    # Both populated by fwd Pass B in the same lane-0 K-loop that writes
    # pool_recv_token / pool_k_slot / per_token_remaining; consumed by the
    # backward path (no fwd consumer). Cost: ~512 KB + ~128 KB at production
    # (T_recv≈32K, K=4).
    recv_token_to_slots: torch.Tensor          # [T_recv, num_topk] int32 — (r, k) → pool slot, -1 for non-local k
    k_local_count: torch.Tensor                # [T_recv] int32 — write-once K_local mirror (per_token_remaining is decremented to 0 by kernel Y)

    total_tiles: int
    tile_m: int
    dispatch_seq: int


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using NVLink)
        - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)
        - low-latency all-to-all (dispatch and combine, using RDMA)

    Attributes:
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
        runtime: the C++ runtime.
    """

    num_sms: int = 20

    def __init__(self,
                 group: Optional[dist.ProcessGroup],
                 num_nvl_bytes: int = 0,
                 num_rdma_bytes: int = 0,
                 allow_mnnvl: bool = False,
                 use_fabric: bool = False,
                 explicitly_destroy: bool = False,
                 enable_shrink: bool = False,
                 comm: Optional["mpi4py.MPI.Comm"] = None) -> None:  # noqa: F821
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode RDMA communication.
            allow_mnnvl: whether to allow MNNVL
            use_fabric: whether to use fabric API for memory buffers.
            enable_shrink: whether to enable shrink mode. The enable mode allocates a mask buffer to support masking ranks dynamically.
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
            comm: the `mpi4py.MPI.Comm` communicator to use in case the group parameter is absent.
        """
        check_nvlink_connections(group)

        # Initialize the CPP runtime
        if group is not None:
            self.rank = group.rank()
            self.group = group
            self.group_size = group.size()

            def all_gather_object(obj):
                object_list = [None] * self.group_size
                dist.all_gather_object(object_list, obj, group)
                return object_list
        elif comm is not None:
            self.rank = comm.Get_rank()
            self.group = comm
            self.group_size = comm.Get_size()

            def all_gather_object(obj):
                return comm.allgather(obj)
        else:
            raise ValueError("Either 'group' or 'comm' must be provided.")
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.explicitly_destroy = explicitly_destroy
        self.enable_shrink = enable_shrink
        self.runtime = stream_ep_cpp.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, explicitly_destroy,
                                          enable_shrink, use_fabric)

        # Synchronize device IDs
        local_device_id = self.runtime.get_local_device_id()
        device_ids = all_gather_object(local_device_id)

        # Synchronize IPC handles
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        ipc_handles = all_gather_object(local_ipc_handle)

        # Synchronize NVSHMEM unique IDs
        root_unique_id = None
        if self.runtime.get_num_rdma_ranks() > 1:
            # Enable IBGDA
            os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'

            # Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
            self.nvshmem_qp_depth = int(os.environ.get('NVSHMEM_QP_DEPTH', '1024'))
            os.environ['NVSHMEM_QP_DEPTH'] = str(self.nvshmem_qp_depth)

            # Reduce gpu memory usage
            # 6 default teams + 1 extra team
            os.environ['NVSHMEM_MAX_TEAMS'] = '7'
            # Disable NVLink SHArP
            os.environ['NVSHMEM_DISABLE_NVLS'] = '1'
            # NOTES: NVSHMEM initialization requires at least 256 MiB
            os.environ['NVSHMEM_CUMEM_GRANULARITY'] = f'{2 ** 29}'

            if not allow_mnnvl:
                # Disable multi-node NVLink detection
                os.environ['NVSHMEM_DISABLE_MNNVL'] = '1'

            # Synchronize using the root ID
            if self.runtime.get_rdma_rank() == 0:
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()
            nvshmem_unique_ids = all_gather_object(root_unique_id)
            root_unique_id = nvshmem_unique_ids[self.runtime.get_root_rdma_rank(True)]

        # Make CPP runtime available
        self.runtime.sync(device_ids, ipc_handles, root_unique_id)
        assert self.runtime.is_available()

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.

        """

        assert self.explicitly_destroy, '`explicitly_destroy` flag must be set'

        self.runtime.destroy()
        self.runtime = None

    @staticmethod
    def is_sm90_compiled():
        return stream_ep_cpp.is_sm90_compiled()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        assert new_num_sms % 2 == 0, 'The SM count must be even'
        Buffer.num_sms = new_num_sms

    def get_local_buffer_tensor(self,
                                dtype: torch.dtype,
                                size: Optional[torch.Size] = None,
                                offset: int = 0,
                                use_rdma_buffer: bool = False) -> torch.Tensor:
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            dtype: the data type (PyTorch `dtype`) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            offset: the offset of the beginning element.
            use_rdma_buffer: whether to return the RDMA buffer.
        """
        tensor = self.runtime.get_local_buffer_tensor(dtype, offset, use_rdma_buffer)
        if size is None:
            return tensor

        assert tensor.numel() >= size.numel()
        return tensor[:size.numel()].view(size)

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 24, 256, 6, 128),
            4: Config(Buffer.num_sms, 6, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 36, 288, 20, 128),
            24: Config(Buffer.num_sms, 32, 288, 8, 128),
            32: Config(Buffer.num_sms, 32, 288, 8, 128),
            48: Config(Buffer.num_sms, 32, 288, 8, 128),
            64: Config(Buffer.num_sms, 32, 288, 8, 128),
            96: Config(Buffer.num_sms, 20, 480, 12, 128),
            128: Config(Buffer.num_sms, 20, 560, 12, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    def _assert_intranode_only(self) -> None:
        """Streaming dispatch / combine are intranode-only today."""
        if self.runtime.get_num_rdma_ranks() > 1:
            raise NotImplementedError(
                "Internode streaming path is not yet implemented."
            )

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 10, 256, 6, 128),
            4: Config(Buffer.num_sms, 9, 256, 6, 128),
            8: Config(Buffer.num_sms, 4, 256, 6, 128),
            16: Config(Buffer.num_sms, 4, 288, 12, 128),
            24: Config(Buffer.num_sms, 1, 288, 8, 128),
            32: Config(Buffer.num_sms, 1, 288, 8, 128),
            48: Config(Buffer.num_sms, 1, 288, 8, 128),
            64: Config(Buffer.num_sms, 1, 288, 8, 128),
            96: Config(Buffer.num_sms, 1, 480, 8, 128),
            128: Config(Buffer.num_sms, 1, 560, 8, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    # noinspection PyTypeChecker
    def dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 topk_idx: torch.Tensor,
                 topk_weights: torch.Tensor,
                 is_token_in_rank: torch.Tensor,
                 num_experts: int,
                 *,
                 expert_alignment: int = 1,
                 tile_m: int = 128,
                 dispatch_seq: int = 1,
                 config: Optional[Config] = None) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
                  'StreamingHandle', EventHandle]:
        """Streaming-MoE pool-layout dispatch (intranode).

        Runs on ``torch.cuda.current_stream()``. The caller is expected to wrap
        this call in ``with torch.cuda.stream(dispatch_stream):`` so dispatch's
        kernels and allocations land on ``dispatch_stream``. Cross-stream
        consumers (kernel A, kernel Y, combine sender) wait on the returned
        ``metadata_done_event`` to safely read metadata tensors
        (``expert_pool_block_offset``, ``pool_recv_token``, ``pool_topk_weight``,
        etc.) without serializing against the dispatch main kernel — preserving
        the per-tile dispatch→A streaming overlap.

        Two kernels + one host sync per layer: a fused metadata kernel
        (``streaming_dispatch_metadata`` — cross-rank count exchange + receiver
        metadata + per-tile arrays), a host poll on
        ``{moe_recv_counter, moe_recv_expert_counter, streaming_total_tiles}``,
        and the dispatch main kernel (pool-layout receiver). The returned
        ``metadata_done_event`` is recorded between the two kernels so consumer
        streams can read metadata tensors without serializing against dispatch
        main. See ``csrc/deep_ep.cpp:intranode_dispatch`` for the full sequence.

        Arguments:
            x: tokens to dispatch, ``[num_tokens, hidden]`` bf16.
            topk_idx: ``[num_tokens, num_topk]`` int64 expert indices (-1 sentinel).
            topk_weights: ``[num_tokens, num_topk]`` float32 expert weights.
            is_token_in_rank: ``[num_tokens, num_ranks]`` bool routing bitmap.
            num_experts: total expert count across all ranks.
            expert_alignment: alignment for per-expert receive counts (default 1).
            tile_m: pool block size (default 128).
            dispatch_seq: monotonic int64 release-stamp on tile_ready.

        Returns:
            recv: ``handle.pool`` (Tensor).
            handle: ``StreamingHandle`` carrying pool tensors + per-tile metadata
                + recv-token-indexed combine inputs.
            metadata_done_event: ``EventHandle`` recorded between
                ``tile_arrays_init`` and the dispatch main kernel. Use as
                ``compute_a_stream.wait_event(metadata_done_event)`` (or the
                equivalent ``metadata_done_event.wait(compute_a_stream)``) to
                make a consumer stream see metadata-tensor writes without
                serializing against dispatch main.
        """
        config = self.get_dispatch_config(self.group_size) if config is None else config
        self._assert_intranode_only()

        out = self.runtime.intranode_dispatch(
            x, topk_idx, topk_weights, is_token_in_rank,
            num_experts, expert_alignment, tile_m, dispatch_seq,
            config,
        )

        handle = StreamingHandle(
            pool=out.pool,
            pool_topk_weight=out.pool_topk_weight,
            pool_recv_token=out.pool_recv_token,
            pool_k_slot=out.pool_k_slot,
            send_head=out.send_head,
            is_token_in_rank=is_token_in_rank,
            rank_prefix_matrix=out.rank_prefix_matrix,
            channel_prefix_matrix=out.channel_prefix_matrix,
            recv_channel_prefix_matrix=out.recv_channel_prefix_matrix,
            expert_frequency=out.expert_frequency,
            expert_pool_block_offset=out.expert_pool_block_offset,
            base_pool=out.base_pool,
            seen_per_substream=out.seen_per_substream,
            tile_id_to_expert=out.tile_id_to_expert,
            pool_arrival_target=out.pool_arrival_target,
            tile_ready=out.tile_ready,
            a_ready=out.a_ready,
            per_token_remaining=out.per_token_remaining,
            compute_done_per_token=out.compute_done_per_token,
            o=out.o,
            recv_token_to_slots=out.recv_token_to_slots,
            k_local_count=out.k_local_count,
            total_tiles=out.total_tiles,
            tile_m=tile_m,
            dispatch_seq=dispatch_seq,
        )

        return out.pool, handle, out.metadata_done_event

    # noinspection PyTypeChecker
    def dispatch_grads(self, handle: 'StreamingHandle', dL_dy: torch.Tensor,
                       *,
                       dispatch_seq: Optional[int] = None,
                       config: Optional[Config] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward dispatch: ship ``dL/dy[t]`` origin → expert ranks along the
        same routing as forward dispatch, write K times into a pool-shaped
        ``dL_do_pool`` using the slot lookup persisted in ``handle.recv_token_to_slots``.

        No metadata kernel, no host poll — reuses the routing already captured
        in ``handle``. Only the small channel-control memset + cross-rank
        barrier (~µs) runs on the host side before the data-ship kernel
        launches. Runs on ``torch.cuda.current_stream()``.
        """
        config = self.get_dispatch_config(self.group_size) if config is None else config
        self._assert_intranode_only()
        seq = handle.dispatch_seq if dispatch_seq is None else dispatch_seq

        TK_padded = handle.pool.shape[0]
        num_topk = handle.recv_token_to_slots.shape[1]
        num_local_experts = handle.expert_frequency.shape[0]
        num_experts = num_local_experts * self.group_size

        dL_do_pool, bwd_y_ready = self.runtime.intranode_dispatch_grads(
            dL_dy,
            handle.is_token_in_rank,
            handle.recv_token_to_slots,
            handle.base_pool,
            handle.seen_per_substream,
            handle.pool_arrival_target,
            handle.rank_prefix_matrix,
            num_experts,
            num_topk,
            handle.tile_m,
            TK_padded,
            seq,
            config,
        )
        return dL_do_pool, bwd_y_ready

    # noinspection PyTypeChecker
    def combine(self, x: torch.Tensor, handle: 'StreamingHandle',
                *,
                combine_seq: int = 1,
                config: Optional[Config] = None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """Combine (reduce) tokens back to source ranks. Takes the ``StreamingHandle``
        returned by ``Buffer.dispatch``. Intranode only.

        Runs on ``torch.cuda.current_stream()``. Caller manages stream placement.

        The combine sender's per-warp send loop spins on
        ``handle.compute_done_per_token[r] >= combine_seq`` before reading
        ``x[r]`` for each token ``r`` it owns — kernel Y release-stores
        ``combine_seq`` into the same address once all ``K_local(r)``
        contributions to ``x[r]`` (= ``handle.o[r]`` in production) have
        landed. Caller threads the same int through ``Buffer.dispatch``
        (``dispatch_seq``), kernel A / Y (``compute_seq``, ``combine_seq``),
        and this call so all four release/acquire pairs key off one layer-
        monotonic ID.

        The per-(r, k) topk-weight payload is loaded via
        ``recv_token_to_slots[r, k] → pool_topk_weight[slot]`` (with 0 for
        non-local k). Same wire format as backward ``combine_grads``; the
        underlying kernel is shared.
        """
        config = self.get_combine_config(self.group_size) if config is None else config
        self._assert_intranode_only()

        return self.runtime.intranode_combine(
            x, handle.pool_topk_weight, handle.recv_token_to_slots,
            handle.rank_prefix_matrix,
            handle.recv_channel_prefix_matrix, handle.send_head,
            handle.compute_done_per_token, combine_seq,
            config)

    # noinspection PyTypeChecker
    def combine_grads(self, dL_dx_per_r: torch.Tensor, handle: 'StreamingHandle',
                      weight_grads: torch.Tensor,
                      bwd_compute_done_per_token: torch.Tensor,
                      *,
                      dispatch_seq: Optional[int] = None,
                      config: Optional[Config] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward combine_grads: ship ``dL/dx_per_r[r, :H]`` expert → origin and
        reduce K contributions per source token into ``dL/dx[t, :H]``, plus
        scatter weight grads into ``dL/dtopk_weights[t, :K]``.

        Underlying kernel is the same `combine_main_kernel` used by fwd
        ``Buffer.combine``; per-direction differences are entirely in args
        (different per-slot weight tensor, different gate variable, different
        outputs, no biases).

        Runs on ``torch.cuda.current_stream()``.

        Args:
            dL_dx_per_r: ``[T_recv, H]`` bf16 — per-recv-token gradient produced
                by ``kernel_a_bwd``'s atomic-scatter epilogue.
            handle: the ``StreamingHandle`` from ``Buffer.dispatch``.
            weight_grads: ``[TK_padded]`` fp32 — per-pool-slot weight gradient
                produced by ``kernel_y_bwd``'s ``dL/dweight``
                dot-product epilogue.
            bwd_compute_done_per_token: ``[T_recv]`` int64 release-stamp
                array fired by ``kernel_a_bwd``'s per-token
                "stripe-done" epilogue.
            dispatch_seq: monotonic int the entire layer threads through;
                defaults to ``handle.dispatch_seq``.

        Returns:
            (dL_dx, dL_dtopk_weights):
                dL_dx: ``[num_tokens, H]`` bf16
                dL_dtopk_weights: ``[num_tokens, num_topk]`` fp32
        """
        config = self.get_combine_config(self.group_size) if config is None else config
        self._assert_intranode_only()
        seq = handle.dispatch_seq if dispatch_seq is None else dispatch_seq

        return self.runtime.intranode_combine(
            dL_dx_per_r, weight_grads, handle.recv_token_to_slots,
            handle.rank_prefix_matrix,
            handle.recv_channel_prefix_matrix, handle.send_head,
            bwd_compute_done_per_token, seq,
            config)

    # noinspection PyTypeChecker
    def internode_dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           handle: Optional[Tuple] = None,
                           num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                           is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                           topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None, expert_alignment: int = 1,
                           num_worst_tokens: int = 0, config: Optional[Config] = None) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
            Optional[torch.Tensor], List[int], Tuple]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.

        Runs on ``torch.cuda.current_stream()``. Caller manages stream placement.
        """
        assert config is not None

        # Launch the kernel with cached or non-cached mode
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            is_token_in_rank, \
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, \
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                recv_src_meta, send_rdma_head, send_nvl_head = handle
            num_recv_tokens = recv_src_meta.size(0)
            num_rdma_recv_tokens = send_nvl_head.size(0)
            recv_x, _, _, _, _, _, _, _, _, _, _, _, _ = self.runtime.internode_dispatch(
                x, topk_idx, topk_weights, None, None, is_token_in_rank, None, num_recv_tokens, num_rdma_recv_tokens,
                rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
                expert_alignment, num_worst_tokens, config)
            return recv_x, None, None, None, None
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, \
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, \
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                recv_src_meta, send_rdma_head, send_nvl_head = self.runtime.internode_dispatch(
                x, topk_idx, topk_weights,
                num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert,
                0, 0, None, None, None, None,
                expert_alignment, num_worst_tokens, config)
            handle = (is_token_in_rank, rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, recv_rdma_channel_prefix_matrix,
                      recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, recv_src_meta, send_rdma_head,
                      send_nvl_head)
            return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle

    # noinspection PyTypeChecker
    def internode_combine(self, x: torch.Tensor, handle: Union[tuple, list],
                          topk_weights: Optional[torch.Tensor] = None,
                          bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                          config: Optional[Config] = None) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.

        Runs on ``torch.cuda.current_stream()``. Caller manages stream placement.
        """
        assert config is not None

        # Unpack handle and bias
        is_combined_token_in_rank, \
            _, _, \
            rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix, gbl_rank_prefix_sum, \
            src_meta, send_rdma_head, send_nvl_head = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        return self.runtime.internode_combine(
            x, topk_weights, bias_0, bias_1, src_meta,
            is_combined_token_in_rank, rdma_channel_prefix_matrix,
            rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
            send_rdma_head, send_nvl_head, config)

