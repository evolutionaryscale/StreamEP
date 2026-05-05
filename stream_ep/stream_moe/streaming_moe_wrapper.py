"""MVP nn.Module wrapper exposing the streaming-MoE pipeline as a SparseMoEBlock backend.

Throughput-only — random expert init, no checkpoint loading, no DP_REPLICATE
gradient sync. Goal is to plug ``stream_moe_func`` into
``projects/moe_benchmark/run_benchmark.py`` so the kernel-bounded numbers from
the streaming pipeline can be compared head-to-head with sonicmoe / scattermoe
on the same harness without touching the rest of the model.

What's intentionally hacky:
- Expert weights are random ``nn.Parameter`` in streaming_moe's native packed
  layout (interleaved gate/up). No checkpoint-load conversion path.
- No DP_REPLICATE gradient all-reduce. The wrapper trains "wrong" across DP
  but the missing collective is off the timed path; throughput is identical.
- One process-global Buffer + StreamHolder, cached by EP group id. The first
  layer pays the IPC-slab allocation cost, the rest hit the cache.

What is NOT hacky (must stay correct for perf to be honest):
- No per-forward weight repacking.
- ``dispatch_seq`` is a monotonic class counter so cross-layer + cross-iter
  uniqueness is guaranteed (CLAUDE.md "Architectural constraints" rule 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_ep import Buffer as DeepEPBuffer
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from evolutionaryscale.models.moe.streaming_moe.streaming_moe import (
    StreamHolder,
    make_streams,
    stream_moe_func,
)
from evolutionaryscale.utils.distributed import (
    Axes,
    device_mesh_initialized,
    get_device_mesh,
    parallelism_initialized,
)


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


# Matches the new sweet-spot default landed in 0dd99b55bc — see
# `bench_pipeline.py` comment for the rationale.
NUM_SMS = 80

# Process-global cache. The Buffer's IPC slabs are heavy and meant to be
# shared across all streaming_moe layers in a model.
_RUNTIME_CACHE: dict[int, tuple[DeepEPBuffer, StreamHolder]] = {}


def _make_buffer(group, num_sms: int, hidden_bytes: int) -> DeepEPBuffer:
    DeepEPBuffer.set_num_sms(num_sms)
    nvl_bytes = rdma_bytes = 0
    for cfg in (
        DeepEPBuffer.get_dispatch_config(group.size()),
        DeepEPBuffer.get_combine_config(group.size()),
    ):
        nvl_bytes = max(
            cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()), nvl_bytes
        )
        rdma_bytes = max(
            cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()), rdma_bytes
        )
    return DeepEPBuffer(
        group, nvl_bytes, rdma_bytes, num_qps_per_rank=DeepEPBuffer.num_sms
    )


class StreamingMoEWrapper(nn.Module):
    """SparseMoEBlock-compatible wrapper around ``stream_moe_func``.

    Exposes the fields ``SparseMoEBlock`` accesses on ``self.moe``:
    - ``.gate`` (router ``nn.Linear``) for the input dtype cast.
    - ``.router_logits`` for aux-loss collection.
    - ``.forward(hidden_states)`` accepting ``(B, S, H)`` or ``(T, H)``.
    - ``.post_init()`` (no-op at MVP).
    - ``.get_params_to_ignore_sharding()`` so FSDP doesn't shard EP weights.
    """

    # Monotonic across all instances + all calls. Must increase across every
    # call using the same Buffer; the per-token combine gate spins on
    # ``compute_done_per_token[r] >= dispatch_seq`` and would latch onto a
    # prior call's stamp if reused.
    _dispatch_seq_counter = 0

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        # ===== EP plumbing — streaming_moe is fundamentally an EP path =====
        ep_axis = getattr(config, "ep_axis", Axes.EP.value)
        if ep_axis is None:
            raise ValueError(
                "StreamingMoEWrapper requires an EP axis; got ep_axis=None"
            )
        if not (device_mesh_initialized() and parallelism_initialized(Axes(ep_axis))):
            raise RuntimeError(
                "StreamingMoEWrapper requires an initialized device mesh with "
                f"axis={ep_axis!r}"
            )
        self._ep_axis = ep_axis
        device_mesh = get_device_mesh()
        self._ep_group = device_mesh.get_group(ep_axis)
        self.ep_size = device_mesh[ep_axis].size()
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"ep_size ({self.ep_size})"
            )
        self.E_local = self.num_experts // self.ep_size

        # ===== Router (full, replicated; same convention as SonicMoE) =====
        self._router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # ===== Expert weights, in streaming_moe's native packed layout =====
        # ``w1_local[..., 2i, :]`` is gate-row-i and ``w1_local[..., 2i+1, :]``
        # is up-row-i (interleaved at the row level — quack's GemmGated epilogue
        # pairs adjacent N-elements). NO per-forward repacking.
        H, I, E = self.hidden_size, self.intermediate_size, self.E_local
        if (2 * I) % 32 != 0:
            raise ValueError(
                f"GemmGatedSm90 requires 2*I divisible by 32; got 2*I={2 * I}"
            )
        self.w1_local = nn.Parameter(torch.empty(E, 2 * I, H, dtype=torch.bfloat16))
        self.w2_local = nn.Parameter(torch.empty(E, H, I, dtype=torch.bfloat16))
        nn.init.normal_(self.w1_local, std=0.02)
        nn.init.normal_(self.w2_local, std=0.02)

        # Lazy runtime — built on first forward when the device is known.
        self._buffer: DeepEPBuffer | None = None
        self._streams: StreamHolder | None = None
        self._router_logits: torch.Tensor | None = None

    # ===== Hooks SparseMoEBlock / FSDP plumbing reads =====

    @property
    def gate(self) -> nn.Linear:
        return self._router

    @property
    def router_logits(self) -> torch.Tensor:
        assert self._router_logits is not None, "router_logits not yet computed"
        return self._router_logits

    def post_init(self) -> None:
        # No-op; benchmark harness calls prepare_expert_parallel_moe directly.
        pass

    def _compute_routing_weights(self, hidden_states: torch.Tensor):
        """Compute (router_logits, routing_weights, selected_experts) — same
        signature as `SonicMoEWrapper._compute_routing_weights` so the
        benchmark's `force_uniform_routing` monkey-patch can override it.

        For the Replicate router weight: pass ``grad_placements=[Partial()]``
        when localising so DTensor's autograd inserts the cross-EP all-reduce
        on dW_router in backward. Without it, each rank computes a different
        local dW (different inputs), the optimizer applies divergent updates,
        and the replicas drift after the first step.
        """
        weight = self._router.weight
        if isinstance(weight, DTensor):
            grad_placements = [Partial()] * weight.device_mesh.ndim
            router_w = weight.to_local(grad_placements=grad_placements)
        else:
            router_w = weight
        router_logits = F.linear(hidden_states.to(router_w.dtype), router_w)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.top_k > 1:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routing_weights = routing_weights.to(hidden_states.dtype)
        return router_logits, routing_weights, selected_experts

    def reset_parameters(self) -> None:
        """Re-init expert weights after `to_empty()` in materialize_sync_parameters.
        Without this, the harness's `to_empty(device)` zaps the std=0.02 init we
        did at __init__ and replaces it with `torch.empty()` garbage; first few
        forward steps then accumulate into NaNs, and downstream routing /
        dispatch breaks. Plain `nn.init.normal_` works on DTensor params too
        (operates on the local shard).
        """
        nn.init.normal_(self.w1_local, std=0.02)
        nn.init.normal_(self.w2_local, std=0.02)
        # Router is an nn.Linear, which already has its own reset_parameters
        # (kaiming_uniform on the weight). Leave it to the standard hook.

    def get_params_to_ignore_sharding(self):
        return {self._router.weight, self.w1_local, self.w2_local}

    def prepare_expert_parallel_moe(self, device_mesh: DeviceMesh) -> None:
        """DTensor-wrap params with Replicate(router) / Shard(0)(experts) on the
        EP axis. Mirrors SonicMoEWrapper.prepare_expert_parallel_moe but
        WITHOUT the DP_REPLICATE post-accumulate-grad hooks — MVP doesn't sync
        DP grads, the missing all-reduce is off the timed path.
        """
        # Outer replicate axes (e.g. DP_REPLICATE on a 2D HSDP mesh) come
        # before the EP shard placement, mirroring SonicMoE / ScatterMoE.
        placements = [Replicate()] if device_mesh.ndim > 1 else []
        self._router.weight = nn.Parameter(
            DTensor.from_local(
                self._router.weight,
                device_mesh=device_mesh,
                placements=placements + [Replicate()],
                run_check=False,
            )
        )
        self.w1_local = nn.Parameter(
            DTensor.from_local(
                self.w1_local,
                device_mesh=device_mesh,
                placements=placements + [Shard(0)],
                run_check=False,
            )
        )
        self.w2_local = nn.Parameter(
            DTensor.from_local(
                self.w2_local,
                device_mesh=device_mesh,
                placements=placements + [Shard(0)],
                run_check=False,
            )
        )

    # ===== Forward =====

    def _ensure_runtime(self, device: torch.device) -> None:
        key = id(self._ep_group)
        cached = _RUNTIME_CACHE.get(key)
        if cached is not None:
            self._buffer, self._streams = cached
            return
        hidden_bytes = self.hidden_size * 2  # bf16
        buffer = _make_buffer(self._ep_group, NUM_SMS, hidden_bytes)
        streams = make_streams(device=device)
        _RUNTIME_CACHE[key] = (buffer, streams)
        self._buffer, self._streams = buffer, streams

    @torch.compiler.disable
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            T = original_shape[0] * original_shape[1]
        else:
            T = original_shape[0]
        x = hidden_states.reshape(T, self.hidden_size)
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x = x.contiguous()
        device = x.device

        self._ensure_runtime(device)
        assert self._buffer is not None and self._streams is not None

        # ---- Routing ----
        # Factored into a method so the benchmark's `force_uniform_routing`
        # monkey-patch (benchmark.py:303-331) can override it for fair
        # comparisons against sonicmoe / scattermoe.
        router_logits, topk_weights, topk_idx = self._compute_routing_weights(x)
        self._router_logits = router_logits
        topk_weights = topk_weights.to(torch.float32).contiguous()
        topk_idx = topk_idx.to(torch.int64).contiguous()

        # ---- is_token_in_rank: (T, ep_size) bool, vectorised ----
        rank_idx = topk_idx // self.E_local  # (T, K)
        ranks = torch.arange(self.ep_size, device=device).view(1, 1, -1)
        is_token_in_rank = (rank_idx.unsqueeze(-1) == ranks).any(dim=1)

        # ---- Streaming MoE call ----
        StreamingMoEWrapper._dispatch_seq_counter += 1
        seq = StreamingMoEWrapper._dispatch_seq_counter
        # Cast fp32 master weights → bf16 for the streaming kernels. quack /
        # streaming_moe_a require bf16 (or fp16) inputs — the fp32 storage
        # is only for the optimizer-side master weights (Adam quantization
        # error stays out of the params, see `__init__` comment).
        w1l = _local(self.w1_local).to(torch.bfloat16)
        w2l = _local(self.w2_local).to(torch.bfloat16)
        out = stream_moe_func(
            self._buffer,
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            w1l,
            w2l,
            streams=self._streams,
            num_experts=self.num_experts,
            dispatch_seq=seq,
            tile_n_a=128,
            tile_n_y=128,
        )
        return out.reshape(original_shape)
