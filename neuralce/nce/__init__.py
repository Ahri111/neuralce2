"""Neural cluster expansion (NeuralCE).

Layer 0 is the cluster expansion evaluated as a sliding window: for every
cluster a map of where it occurs on the lattice (``general.local_cluster_features``,
whose site mean equals icet's cluster vector to 1e-9).  Graph convolutions on
the lattice (``general.make_forward``) keep the maps site-resolved; the energy
per atom is the site average of the result.

general               layer 0 from icet, neighbour shells, forward model, training utilities
deep                  deeper convolution stack (shells built from icet pair orbits)
conv / kernels / lattice / features   roll-based convolution on a cubic fcc grid (PtNi)
spin_features         spin channels (Heisenberg pairs per site) for magnetic alloys
gpu_layer0 / _tiled / _implicit       layer-0 tables on the GPU (legacy / tiled / template-only)
local_update / _implicit              receptive-field local energy update for Monte Carlo
mc_model              load a trained model.pkl and build the MC machinery for any supercell
mcmc                  single-T, multi-T and parallel-tempering drivers (``neuralce-mc --framework nce``)
"""
