from kernel_launch import launch
from grakel.kernels import ShortestPath
from grakel.kernels import NeighborhoodSubgraphPairwiseDistance
from grakel.kernels import WeisfeilerLehman
from grakel.kernels.graphlet_sampling import GraphletSampling


launch(ShortestPath)
launch(NeighborhoodSubgraphPairwiseDistance)
launch(WeisfeilerLehman)
launch(GraphletSampling)

