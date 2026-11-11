"""GPU cluster expansion in the orthonormal (icet) basis.

classical_ce_gpu      GPUClusterExpansion: icet .ce file -> GPU tables, energy and swap kernels
ce_gpu_tiled          translation-template tables (O(N) int32 rows)
ce_gpu_implicit       template-only kernels: GPU memory independent of the cell size
pt_mcmc_ce            parallel-tempering engine (jitted lax.scan, vmapped replicas)
icet_mc / gpu_mc      single-T, multi-T and PT drivers used by ``neuralce-mc --framework icet``
primitive_no_idealize primitive-cell detection without idealising the structure
train_ce              Lasso / ARDR fit of an icet cluster expansion from a structure set
spin_pairs            Heisenberg pair features of the Fe-Ni-Cr spin cluster expansion
"""
