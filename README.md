# Single Chain Slide Ring

Perform single-chain slide-ring polymer simulation for obtaining stress-strain curves of slide-ring networks. Virtual chains need to be adjusted to match the the real slide-ring network.

1. Initialization of chain conformations with Gaussian distribution
2. Rings are attached to elastic background with tunable virtual chains
3. Random shuffling of number of monomers between rings with Metropolis algorithm based on free energy
4. Positions of rings are solved by balancing the forces
5. Ensemble average of stress tensor with Monte Carlo sampling
6. Affine deformation of the elastic background
7. Visulization of slide-ring chains
