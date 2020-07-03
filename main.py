from kernel_launch import launch
from grakel.kernels import ShortestPath
from grakel.kernels import NeighborhoodSubgraphPairwiseDistance
from grakel.kernels import WeisfeilerLehman
from grakel.kernels.graphlet_sampling import GraphletSampling
from grakel.kernels.random_walk import RandomWalk


launch(ShortestPath)
launch(NeighborhoodSubgraphPairwiseDistance)
launch(WeisfeilerLehman)
launch(GraphletSampling)
launch(RandomWalk)

