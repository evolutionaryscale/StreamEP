# rdma-core headers

Unmodified headers from [rdma-core](https://github.com/linux-rdma/rdma-core) v65.0 (taken from conda-forge `rdma-core-65.0-h7b4f2e6_0`). NVSHMEM's IBGDA device headers and `csrc/kernels/configs.cuh` include `<infiniband/mlx5dv.h>`; these are that header and everything it pulls in from rdma-core. The remaining includes (`rdma/ib_user_verbs.h`, `linux/*.h`) come from the compiler's sysroot kernel headers.

They are vendored instead of installed because the conda `rdma-core` package also installs `libibverbs` / `libmlx5`. At runtime NCCL and NVSHMEM `dlopen` those by name, Python's RPATH resolves the env's copies, and they clash with the host's RDMA stack (MLNX OFED) that the system NCCL net plugin loads, leaving no usable IB devices. The extension only needs the headers; it links neither library.

## Licence

Each header carries its own notice: dual GPL-2.0 / OpenIB.org BSD (`ib_user_ioctl_verbs.h`: `(GPL-2.0 WITH Linux-syscall-note) OR Linux-OpenIB`). StreamEP uses them under the BSD terms reproduced in each file.

## Refreshing

Only needed if a newer NVSHMEM requires definitions these lack. Copy the same files from the matching rdma-core release's `include/infiniband/` (or an installed `rdma-core`), re-check the include closure of `mlx5dv.h`, and update the version above.
