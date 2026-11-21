# OptiX route (RTX GPUs, e.g. RTX PRO 6000)

Cycles' OptiX device traces rays on RT cores with an OptiX module built from PTX, and runs every shading kernel from
the CUDA cubin. These patches let the `nrc` and `exact` kernel trees run that way:

- `kernel.diff` (on a tree from `scripts/kernel_tree.sh`): the OptiX launch parameters carry `KernelNRC`
  (`kernel_nrc` is `kernel_params.nrc`; an OptiX module's own `__constant__` symbols cannot be set by the host), and
  the `ccl_device_hot*` qualifiers of `CYCLES_INLINE_HOT` exist for OptiX too, and the stock random-walk SSS (which
  the OptiX module runs) gets the walk roulette of the CUDA dynamic-fetch walk (`CYCLES_SSS_WALK_RR_*`, nrc only).
- `kernel-exact.diff` (exact tree only, after `kernel.diff`): adds `kernel/integrator/nrc_types.h`, which the
  launch-parameter layout needs; the exact tree has no cache.
- `host.diff` (on the host source after `patches/host.diff`): the same launch-parameter layout; `CYCLES_OPTIX_PTX=<file>`
  loads a per-scene base PTX (optional modules from the same directory); `CYCLES_OPTIX_MAX_REGISTERS=<n>` sets the
  OptiX module register limit; `const_copy_to("kernel_nrc")` fills the launch parameters.

Build: `PATCH2=patches/optix/host.diff OPTIX_ROOT=<OptiX 8 SDK> scripts/build_host.sh ...`, PTX with
`scripts/build_ptx.sh`. See RESULTS-rtxpro6000.md.
