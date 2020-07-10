![foscari](/Users/rr/PycharmProjects/Manifold-Learning-and-Graph-Kernels/foscari.jpg)

# Manifold Learning and Graph Kernels

Third Assignment of the course in Artficial Intelligence held by Prof. Torsello

Bernardi Riccardo - 864018

<div style="page-break-after: always;"></div>

# Index:

[TOC]

<div style="page-break-after: always;"></div>

# 1. Problem Statement

Read [this article](https://www.dsi.unive.it/~atorsell/AI/graph/Unfolding.pdf) presenting a way to improve the disciminative power of graph kernels.

Choose one [graph kernel](https://www.dsi.unive.it/~atorsell/AI/graph/kernels.pdf) among

- Shortest-path Kernel
- Graphlet Kernel
- Random Walk Kernel
- Weisfeiler-Lehman Kernel

Choose one manifold learning technique among

- Isomap
- Diffusion Maps
- Laplacian Eigenmaps
- Local Linear Embedding

Compare the performance of an SVM trained on the given kernel, with or without the manifold learning step, on the following datasets:

- [PPI](https://www.dsi.unive.it/~atorsell/AI/graph/PPI.mat)
- [Shock](https://www.dsi.unive.it/~atorsell/AI/graph/SHOCK.mat)

**Note:** the datasets are contained in Matlab files. The variable G contains a vector of cells, one per graph. The entry am of each cell is the adjacency matrix of the graph. The variable labels, contains the class-labels of each graph. 

NEW I have added zip files with csv versions of the adjacecy matrices of the graphs and of the lavels. the files graphxxx.csv contain the adjaccency matrices, one per file, while the file labels.csv contais all the labels 

- [PPI](https://www.dsi.unive.it/~atorsell/AI/graph/PPI.zip)
- [Shock](https://www.dsi.unive.it/~atorsell/AI/graph/SHOCK.zip)



# 2. Introduction



Protein Protein Interaction dataset



# 3. The Graph Kernel



We are going here to answer these questions:

- what is a kernel and how to create one ? 
- what is a graph kernel? 
- which kernels are available and where ?



### 3.1 What is a kernel

In [machine learning](https://en.wikipedia.org/wiki/Machine_learning), **kernel methods** are a class of algorithms for [pattern analysis](https://en.wikipedia.org/wiki/Pattern_analysis), whose best known member is the [support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine) (SVM). The general task of pattern analysis is to find and study general types of relations (for example [clusters](https://en.wikipedia.org/wiki/Cluster_analysis), [rankings](https://en.wikipedia.org/wiki/Ranking), [principal components](https://en.wikipedia.org/wiki/Principal_components), [correlations](https://en.wikipedia.org/wiki/Correlation), [classifications](https://en.wikipedia.org/wiki/Statistical_classification)) in datasets. For many algorithms that solve these tasks, the data in raw representation have to be explicitly transformed into [feature vector](https://en.wikipedia.org/wiki/Feature_vector) representations via a user-specified *feature map*: in contrast, kernel methods require only a user-specified *kernel*, i.e., a [similarity function](https://en.wikipedia.org/wiki/Similarity_function) over pairs of data points in raw representation.

Kernel methods owe their name to the use of [kernel functions](https://en.wikipedia.org/wiki/Positive-definite_kernel), which enable them to operate in a high-dimensional, *implicit* [feature space](https://en.wikipedia.org/wiki/Feature_space) without ever computing the coordinates of the data in that space, but rather by simply computing the [inner products](https://en.wikipedia.org/wiki/Inner_product) between the [images](https://en.wikipedia.org/wiki/Image_(mathematics)) of all pairs of data in the feature space. This operation is often computationally cheaper than the explicit computation of the coordinates. This approach is called the "**kernel trick**".[[1\]](https://en.wikipedia.org/wiki/Kernel_method#cite_note-1) Kernel functions have been introduced for sequence data, [graphs](https://en.wikipedia.org/wiki/Graph_kernel), text, images, as well as vectors.



The kernel trick avoids the explicit mapping that is needed to get linear [learning algorithms](https://en.wikipedia.org/wiki/Learning_algorithms) to learn a nonlinear function or [decision boundary](https://en.wikipedia.org/wiki/Decision_boundary). For all x![\mathbf {x} ](https://wikimedia.org/api/rest_v1/media/math/render/svg/32adf004df5eb0a8c7fd8c0b6b7405183c5a5ef2) and x′![\mathbf {x'} ](https://wikimedia.org/api/rest_v1/media/math/render/svg/7d14ab6186e99346cb608a30858c3e1580f760e6) in the input space X![{\mathcal {X}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8c7e5461c5286852df4ef652fca7e4b0b63030e9), certain functions k(x,x′)![k(\mathbf {x} ,\mathbf {x'} )](https://wikimedia.org/api/rest_v1/media/math/render/svg/7d02f87329f893c16295074bcfe9d974fb72c4eb) can be expressed as an [inner product](https://en.wikipedia.org/wiki/Inner_product) in another space V![{\mathcal {V}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/47d69f309b6deb2e5008f6130ee11e09bbabd7b6). The function k:X×X→R![k\colon {\mathcal {X}}\times {\mathcal {X}}\to \mathbb {R} ](https://wikimedia.org/api/rest_v1/media/math/render/svg/ecc3c27f7e04f4ce7c0088a69e0b414a74869e3e) is often referred to as a *kernel* or a *[kernel function](https://en.wikipedia.org/wiki/Kernel_function)*. The word "kernel" is used in mathematics to denote a weighting function for a weighted sum or [integral](https://en.wikipedia.org/wiki/Integral).

Certain problems in machine learning have more structure than an arbitrary weighting function k![k](https://wikimedia.org/api/rest_v1/media/math/render/svg/c3c9a2c7b599b37105512c5d570edc034056dd40). The computation is made much simpler if the kernel can be written in the form of a "feature map" φ:X→V![\varphi \colon {\mathcal {X}}\to {\mathcal {V}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/b8484d2e5a2fb0ed38f151079dceaca4a395eca5) which satisfies



The key restriction is that ⟨⋅,⋅⟩V![\langle \cdot ,\cdot \rangle _{\mathcal {V}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/cfaf5152d8c0788782803dc35974e62634f7a635) must be a proper inner product. On the other hand, an explicit representation for φ![\varphi ](https://wikimedia.org/api/rest_v1/media/math/render/svg/33ee699558d09cf9d653f6351f9fda0b2f4aaa3e) is not necessary, as long as V![{\mathcal {V}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/47d69f309b6deb2e5008f6130ee11e09bbabd7b6) is an [inner product space](https://en.wikipedia.org/wiki/Inner_product_space). The alternative follows from [Mercer's theorem](https://en.wikipedia.org/wiki/Mercer's_theorem): an implicitly defined function φ![\varphi ](https://wikimedia.org/api/rest_v1/media/math/render/svg/33ee699558d09cf9d653f6351f9fda0b2f4aaa3e) exists whenever the space X![{\mathcal {X}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8c7e5461c5286852df4ef652fca7e4b0b63030e9) can be equipped with a suitable [measure](https://en.wikipedia.org/wiki/Measure_(mathematics))ensuring the function k![k](https://wikimedia.org/api/rest_v1/media/math/render/svg/c3c9a2c7b599b37105512c5d570edc034056dd40) satisfies [Mercer's condition](https://en.wikipedia.org/wiki/Mercer's_condition).

Mercer's theorem is similar to a generalization of the result from linear algebra that [associates an inner product to any positive-definite matrix](https://en.wikipedia.org/wiki/Positive-definite_matrix#Characterizations). In fact, Mercer's condition can be reduced to this simpler case. If we choose as our measure the [counting measure](https://en.wikipedia.org/wiki/Counting_measure) μ(T)=|T|![\mu (T)=|T|](https://wikimedia.org/api/rest_v1/media/math/render/svg/a44af909e4eab9daa53ba7b9e6901c8df5bf2bc1) for all T⊂X![T\subset X](https://wikimedia.org/api/rest_v1/media/math/render/svg/28eb68af8d21de80992dc26e6ee9b6a99f5c54f9), which counts the number of points inside the set T![T](https://wikimedia.org/api/rest_v1/media/math/render/svg/ec7200acd984a1d3a3d7dc455e262fbe54f7f6e0), then the integral in Mercer's theorem reduces to a summation



If this summation holds for all finite sequences of points (x1,…,xn)![(\mathbf {x} _{1},\dotsc ,\mathbf {x} _{n})](https://wikimedia.org/api/rest_v1/media/math/render/svg/5b4e4c8cc45e704f262c34fda4e3a3fa52754d0e) in X![{\mathcal {X}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8c7e5461c5286852df4ef652fca7e4b0b63030e9) and all choices of n![n](https://wikimedia.org/api/rest_v1/media/math/render/svg/a601995d55609f2d9f5e233e36fbe9ea26011b3b) real-valued coefficients (c1,…,cn)![(c_{1},\dots ,c_{n})](https://wikimedia.org/api/rest_v1/media/math/render/svg/bd731fbc6215cae64c53bf0120c4ebfee01d3f96) (cf. [positive definite kernel](https://en.wikipedia.org/wiki/Positive_definite_kernel)), then the function k![k](https://wikimedia.org/api/rest_v1/media/math/render/svg/c3c9a2c7b599b37105512c5d570edc034056dd40) satisfies Mercer's condition.

Some algorithms that depend on arbitrary relationships in the native space X![{\mathcal {X}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8c7e5461c5286852df4ef652fca7e4b0b63030e9) would, in fact, have a linear interpretation in a different setting: the range space of φ![\varphi ](https://wikimedia.org/api/rest_v1/media/math/render/svg/33ee699558d09cf9d653f6351f9fda0b2f4aaa3e). The linear interpretation gives us insight about the algorithm. Furthermore, there is often no need to compute φ![\varphi ](https://wikimedia.org/api/rest_v1/media/math/render/svg/33ee699558d09cf9d653f6351f9fda0b2f4aaa3e) directly during computation, as is the case with [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machines). Some cite this running time shortcut as the primary benefit. Researchers also use it to justify the meanings and properties of existing algorithms.

Theoretically, a [Gram matrix](https://en.wikipedia.org/wiki/Gram_matrix) K∈Rn×n![\mathbf {K} \in \mathbb {R} ^{n\times n}](https://wikimedia.org/api/rest_v1/media/math/render/svg/5ddf49f743a8541a3c6812638951cc6d13015d07) with respect to {x1,…,xn}![\{\mathbf {x} _{1},\dotsc ,\mathbf {x} _{n}\}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8246b26ae9914d260265dfbea03c55be5e8d00b3) (sometimes also called a "kernel matrix"[[3\]](https://en.wikipedia.org/wiki/Kernel_method#cite_note-3)), where Kij=k(xi,xj)![{\displaystyle K_{ij}=k(\mathbf {x} _{i},\mathbf {x} _{j})}](https://wikimedia.org/api/rest_v1/media/math/render/svg/bd50a73f3c68c1ec4fad86ce50b4c413b22b075e), must be [positive semi-definite (PSD)](https://en.wikipedia.org/wiki/Positive-definite_matrix).[[4\]](https://en.wikipedia.org/wiki/Kernel_method#cite_note-4) Empirically, for machine learning heuristics, choices of a function k![k](https://wikimedia.org/api/rest_v1/media/math/render/svg/c3c9a2c7b599b37105512c5d570edc034056dd40) that do not satisfy Mercer's condition may still perform reasonably if k![k](https://wikimedia.org/api/rest_v1/media/math/render/svg/c3c9a2c7b599b37105512c5d570edc034056dd40) at least approximates the intuitive idea of similarity.[[5\]](https://en.wikipedia.org/wiki/Kernel_method#cite_note-5) Regardless of whether k![k](https://wikimedia.org/api/rest_v1/media/math/render/svg/c3c9a2c7b599b37105512c5d570edc034056dd40) is a Mercer kernel, k![k](https://wikimedia.org/api/rest_v1/media/math/render/svg/c3c9a2c7b599b37105512c5d570edc034056dd40) may still be referred to as a "kernel".



### 3.2 What is a Graph kernel

In [structure mining](https://en.wikipedia.org/wiki/Structure_mining), a domain of learning on structured data objects in [machine learning](https://en.wikipedia.org/wiki/Machine_learning), a **graph kernel** is a [kernel function](https://en.wikipedia.org/wiki/Positive-definite_kernel) that computes an [inner product](https://en.wikipedia.org/wiki/Inner_product_space) on [graphs](https://en.wikipedia.org/wiki/Graph_(abstract_data_type)).[[1\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-Vishwanathan-1) Graph kernels can be intuitively understood as functions measuring the similarity of pairs of graphs. They allow [kernelized](https://en.wikipedia.org/wiki/Kernel_trick) learning algorithms such as [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine) to work directly on graphs, without having to do [feature extraction](https://en.wikipedia.org/wiki/Feature_extraction) to transform them to fixed-length, real-valued [feature vectors](https://en.wikipedia.org/wiki/Feature_vector). They find applications in [bioinformatics](https://en.wikipedia.org/wiki/Bioinformatics), in [chemoinformatics](https://en.wikipedia.org/wiki/Chemoinformatics) (as a type of [molecule kernels](https://en.wikipedia.org/wiki/Molecule_kernel)[[2\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-Ralaivola2005-2)), and in [social network analysis](https://en.wikipedia.org/wiki/Social_network_analysis).[[1\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-Vishwanathan-1)

Concepts of graph kernels have been around since the 1999, when D. Haussler[[3\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-3) introduced convolutional kernels on discrete structures. The term graph kernels was more officially coined in 2002 by R. I. Kondor and John Lafferty[[4\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-4) as kernels *on* graphs, i.e. similarity functions between the nodes of a single graph, with the [World Wide Web](https://en.wikipedia.org/wiki/World_Wide_Web) [hyperlink](https://en.wikipedia.org/wiki/Hyperlink) graph as a suggested application. In 2003, Gaertner *et al.*[[5\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-Gaertner-5) and Kashima *et al.*[[6\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-Kashima-6) defined kernels *between* graphs. In 2010, Vishwanathan *et al.* gave their unified framework.[[1\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-Vishwanathan-1) In 2018, Ghosh et al. [[7\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-7) described the history of graph kernels and their evolution over two decades.

An example of a kernel between graphs is the **random walk kernel**[[5\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-Gaertner-5)[[6\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-Kashima-6), which conceptually performs [random walks](https://en.wikipedia.org/wiki/Random_walk) on two graphs simultaneously, then counts the number of [paths](https://en.wikipedia.org/wiki/Path_(graph_theory)) that were produced by *both* walks. This is equivalent to doing random walks on the [direct product](https://en.wikipedia.org/wiki/Tensor_product_of_graphs) of the pair of graphs, and from this, a kernel can be derived that can be efficiently computed.[[1\]](https://en.wikipedia.org/wiki/Graph_kernel#cite_note-Vishwanathan-1)

**Graph isomorphism**

Find a mapping f of the vertices of G1 to the vertices of G2 such that G1 and G2 are identical; i.e. (x,y) is an edge of G1 iff (f(x),f(y)) is an edge of G2. Then f is an isomorphism, and G1 and G2 are called isomorphic

No polynomial-time algorithm is known for graph isomorphism Neither is it known to be NP-complete

**Subgraph isomorphism**

Subgraph isomorphism asks if there is a subset of edges and vertices of G1 that is isomorphic to a smaller graph G2

Subgraph isomorphism is NP-complete

**NP-completeness**

A decision problem C is NP-complete iff
CisinNP
C is NP-hard, i.e. every other problem in NP is reducible to it.

**Problems for the practitioner**

Excessive runtime in worst case
 Runtime may grow exponentially with the number of nodes

For larger graphs with many nodes and for large datasets of graphs, this is an enormous problem

**Graph kernels inspired by concepts from chemoinformatics**

!  

!  

!  

Define three new kernels (Tanimoto, MinMax, Hybrid) for function prediction of chemical compounds

Based on the idea of molecular fingerprints and

Counting labeled paths of depth up to *d* using depth-first search from each possible vertex

**Properties**

!  Tailored for applications in chemical informatics, !  Exploit the small size and
 !  Low average degree of these molecular graphs.

### 3.3 The available kernels



**Principle**

!  Count common walks in two input graphs G and G’
 !  Walks are sequences of nodes that allow repetitions of nodes

**Elegant computation**

!  Walks of length *k* can be computed by looking at the *k*-th power of the adjacency matrix !  Construct direct product graph of G and G'
 !  Count walks in this product graph Gx=(Vx,Ex)
 !  Each walk in the product graph corresponds to one walk in G and G'



**Disadvantages**

!  Runtime problems !  Tottering
 !  'Halting'

**Potential solutions**

!  Fast computation of random walk graph kernels (Vishwanathan et al., NIPS 2006) !  Preventing tottering and label enrichment (Mahe et al., ICML 2004)
 !  Graph kernels based on shortest paths (B. and Kriegel, ICDM 2005)



**Direct computation: O(n****6****)**

**Solution**

!  Cast computation of random walk kernel as Sylvester Equation !  These can be solved in O(n3)

**Vec-Operator**

vec flattens an n x n matrix A into an n2 x1 vector vec(A).
 !  It stacks the columns of the matrix on top of each other, from left to right.

Vec-Operator and Kronecker Products

**Kronecker Product**

!  Product of two matrices A and B
 !  Each element of A is multiplied with the full matrix B:



**Phenomenon of tottering**

!  Walks allow for repetitions of nodes
 !  A walk can visit the same cycle of nodes all over again
 !  Kernel measures similarity in terms of common walks
 !  Hence a small structural similarity can cause a huge kernel value



Subtree Kernel (Ramon and Gaertner, 2004)



**Principle**

!  Compare subtree-like patterns in two graphs
 !  Subtree-like pattern is a subtree that allows for repetitions of nodes and edges (similar to

walk versus path)
 !  ForallpairsofnodesvfromGandufromG‘:

!  Compare u and v via a kernel function
 !  Recursively compare all sets of neighbours of u and v via a kernel function

**Advantages**

!  Richer representation of graph structure than walk-based approach

**Disadvantages**

!  Runtime grows exponentially with the recursion depth of the subtree-like patterns



Graphlet Kernel (B., Petri, et al., MLG 2007)



**Principle**

!  Count subgraphs of limited size *k* in *G* and *G‘
\* !  These subgraphs are referred to as **graphlets** (Przulj, Bioinformatics 2007) !  Define graph kernel that counts isomorphic graphlets in two graphs

**Runtime problems**

!  Pairwise test of isomorphism is expensive !  Number of graphlets scales as O(nk)

**Two solutions on unlabeled graphs**

!  Precompute isomorphisms !  Sample graphlets

**Disadvantage**

!  Same solutions not feasible on labeled graphs







### 3.3 Examples 

examples of some kernels



# 4. The Manifold Technique



Introduction to Manifold Techniques, what is it?



### 4.1 The available Manifold Techniques

explain the choice of a Manifold Techniques

### 4.2 Examples 

examples of some Manifold Techniques





# 5. Comparison

Train an SVM with the kernel chosen applying the manifold technique or not and show the difference



|      | method                  | PPI_score   | SHOCK_score |
| ---: | :---------------------- | :---------- | :---------- |
|    0 | SPK-precomputed-no-RED  | Acc: 49.17% | Acc: 0.0%   |
|    1 | SPK-linear-no-RED       | Acc: 75.14% | Acc: 43.0%  |
|    2 | SPK-rbf-no-RED          | Acc: 62.22% | Acc: 31.5%  |
|    3 | WLK-precomputed-no-RED  | Acc: 42.64% | Acc: 3.0%   |
|    4 | WLK-linear-no-RED       | Acc: 75.56% | Acc: 38.5%  |
|    5 | WLK-rbf-no-RED          | Acc: 47.78% | Acc: 26.0%  |
|    6 | STK-precomputed-no-RED  | Acc: 41.94% | Acc: 1.5%   |
|    7 | STK-linear-no-RED       | Acc: 73.47% | Acc: 42.5%  |
|    8 | STK-rbf-no-RED          | Acc: 67.36% | Acc: 39.5%  |
|    9 | DSGK-precomputed-no-RED | Acc: 36.25% | Acc: 3.5%   |
|   10 | DSGK-linear-no-RED      | Acc: 79.17% | Acc: 42.0%  |
|   11 | DSGK-rbf-no-RED         | Acc: 67.5%  | Acc: 30.0%  |
|   12 | SPK-linear-ISO          | Acc: 62.92% | Acc: 32.0%  |
|   13 | SPK-rbf-ISO             | Acc: 75.83% | Acc: 34.5%  |
|   14 | WLK-linear-ISO          | Acc: 53.47% | Acc: 22.5%  |
|   15 | WLK-rbf-ISO             | Acc: 64.03% | Acc: 33.0%  |
|   16 | STK-linear-ISO          | Acc: 56.81% | Acc: 17.0%  |
|   17 | STK-rbf-ISO             | Acc: 57.22% | Acc: 29.5%  |
|   18 | DSGK-linear-ISO         | Acc: 66.25% | Acc: 23.0%  |
|   19 | DSGK-rbf-ISO            | Acc: 73.19% | Acc: 33.0%  |
|   20 | SPK-linear-LLE          | Acc: 53.33% | Acc: 18.0%  |
|   21 | SPK-rbf-LLE             | Acc: 53.33% | Acc: 18.0%  |
|   22 | WLK-linear-LLE          | Acc: 53.33% | Acc: 19.5%  |
|   23 | WLK-rbf-LLE             | Acc: 53.33% | Acc: 19.5%  |
|   24 | STK-linear-LLE          | Acc: 53.33% | Acc: 11.0%  |
|   25 | STK-rbf-LLE             | Acc: 53.33% | Acc: 12.5%  |
|   26 | DSGK-linear-LLE         | Acc: 53.33% | Acc: 22.0%  |
|   27 | DSGK-rbf-LLE            | Acc: 53.33% | Acc: 22.0%  |
|   28 | SPK-linear-TSNE         | Acc: 56.67% | Acc: 39.5%  |
|   29 | SPK-rbf-TSNE            | Acc: 57.08% | Acc: 41.5%  |
|   30 | WLK-linear-TSNE         | Acc: 70.83% | Acc: 27.5%  |
|   31 | WLK-rbf-TSNE            | Acc: 70.83% | Acc: 41.5%  |
|   32 | STK-linear-TSNE         | Acc: 48.75% | Acc: 15.0%  |
|   33 | STK-rbf-TSNE            | Acc: 53.33% | Acc: 36.5%  |
|   34 | DSGK-linear-TSNE        | Acc: 61.67% | Acc: 25.5%  |
|   35 | DSGK-rbf-TSNE           | Acc: 74.31% | Acc: 33.5%  |



### 4.1 Training without Manifold

Training without Manifold

### 4.2 Training with Manifold

Training with Manifold

### 4.2 Results

Results of Manifold Techniques





# 6. Conclusions



# Bibliography

1. Manifold Learning and Dimensionality Reduction for Data Visualization... - Stefan Kühn - https://www.youtube.com/watch?v=j8080l9Pvic
2. Unsupervised Learning Explained (+ Clustering, Manifold Learning, ...) - https://www.youtube.com/watch?v=-OEgiMH5aok
3. Unfolding Kernel Embeddings of Graphs - https://www.dsi.unive.it/~atorsell/AI/graph/Unfolding.pdf
4. Graph Kernels - https://www.dsi.unive.it/~atorsell/AI/graph/kernels.pdf
5. Manifold Learning - https://scikit-learn.org/stable/modules/manifold.html
6. In-Depth: Manifold Learning - https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
7. What Is Manifold Learning? - https://prateekvjoshi.com/2014/06/21/what-is-manifold-learning/
8. Manifold Learning: The Theory Behind It - https://towardsdatascience.com/manifold-learning-the-theory-behind-it-c34299748fec
9. Introduction to manifold learning - https://onlinelibrary.wiley.com/doi/pdf/10.1002/wics.1222
10. Is Manifold Learning for Toy Data only? - https://www.stat.washington.edu/mmp/Talks/mani-MMDS16.pdf
11. Proximity graphs for clustering and manifold learning - https://papers.nips.cc/paper/2681-proximity-graphs-for-clustering-and-manifold-learning.pdf
12. Manifold Learning - https://indico.in2p3.fr/event/6040/attachments/29587/36427/Manifold_learning.pdf
13. GRAPH CONSTRUCTION FOR MANIFOLD DISCOVERY - https://people.cs.umass.edu/~ccarey/pubs/thesis.pdf
14. Machine Learning on Graphs: A Model and Comprehensive Taxonomy - https://arxiv.org/pdf/2005.03675.pdf
15. Manifold Learning and Spectral Methods - http://mlss2018.net.ar/slides/Pfau-1.pdf
16. Manifold Learning in the Age of Big Data - https://www.stat.washington.edu/mmp/Talks/mani-sppexa19.pdf
17. manifold learning with applications to object recognition - https://people.eecs.berkeley.edu/~efros/courses/AP06/presentations/ThompsonDimensionalityReduction.pdf
18. Representation Learning on Graphs: Methods and Applications - https://www-cs.stanford.edu/people/jure/pubs/graphrepresentation-ieee17.pdf
19. Data Analysis and Manifold Learning (DAML) - http://perception.inrialpes.fr/people/Horaud/Courses/DAML_2011.html
20. Spectral Methods for Dimensionality Reduction - http://cseweb.ucsd.edu/~saul/papers/smdr_ssl05.pdf
21. 􏰁Robust Principal Component Analysis for Computer Vision - http://files.is.tue.mpg.de/black/papers/rpca.pdf
22. K-means Clustering via Principal Component Analysis - http://ranger.uta.edu/~chqding/papers/KmeansPCA1.pdf
23. K-means Clustering & PCA - https://www.inf.ed.ac.uk/teaching/courses/inf2b/labs/learn-lab3.pdf
24. Charting a Manifold - https://papers.nips.cc/paper/2165-charting-a-manifold.pdf
25. Learning High Dimensional Correspondences from Low Dimensional Manifolds - https://repository.upenn.edu/cgi/viewcontent.cgi?article=1131&context=ese_papers
26. Is manifold learning for toy data only?, Marina Meila - https://www.youtube.com/watch?v=ddhbjCLIjho
27. Locally Linear Embedding - https://www.youtube.com/watch?v=scMntW3s-Wk&list=PL_AYx6iB_DjTXmIN126hH2wZc1aGWb0u9
28. A Global Geometric Framework for Nonlinear Dimensionality Reduction - http://www.robots.ox.ac.uk/~az/lectures/ml/tenenbaum-isomap-Science2000.pdf
29. Laplacian Eigenmaps for dimensionality reduction and data representation - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.5888&rep=rep1&type=pdf
30. Non-linear dimension reduction - http://statweb.stanford.edu/~tibs/sta306bfiles/isomap.pdf
31. Isomap - Isometric feature mapping - https://www.cise.ufl.edu/class/cap6617fa17/ISOMAP.pptx.pdf
32. Pattern Search Multidimensional Scaling - https://arxiv.org/pdf/1806.00416.pdf
33. An Introduction to Locally Linear Embedding - https://cs.nyu.edu/~roweis/lle/papers/lleintro.pdf
34. Nonlinear Dimensionality Reduction by Locally Linear Embedding - http://www.robots.ox.ac.uk/~az/lectures/ml/lle.pdf
35. Manifold Learning: The Price of Normalization - http://www.jmlr.org/papers/volume9/goldberg08a/goldberg08a.pdf
36. Dimensionality Estimation, Manifold Learning and Function Approximation using Tensor Voting - http://www.jmlr.org/papers/volume11/mordohai10a/mordohai10a.pdf
37. Riemannian Manifolds - An Introduction to Curvature - https://www.maths.ed.ac.uk/~v1ranick/papers/leeriemm.pdf
38. Adaptive Neighboring Selection Algorithm Based on Curvature Prediction in Manifold Learning - https://arxiv.org/pdf/1704.04050.pdf
39. Nonlinear Manifold Learning Part One: Background, LLE, IsoMap - http://web.mit.edu/6.454/www/www_fall_2003/ihler/slides.pdf
40. Sparse Manifold Clustering and Embedding - http://cis.jhu.edu/~ehsan/Downloads/SMCE-NIPS11-Ehsan.pdf
41. Sampling Methods for the Nystrom Method - http://www.jmlr.org/papers/volume13/kumar12a/kumar12a.pdf
42. On the Nystro ̈m Method for Approximating a Gram Matrix for Improved Kernel-Based Learning - http://www.jmlr.org/papers/volume6/drineas05a/drineas05a.pdf
43. Ensemble Nystro ̈m Method - https://papers.nips.cc/paper/3850-ensemble-nystrom-method.pdf
44. Revisiting the Nystr ̈om method for improved large-scale machine learning - http://proceedings.mlr.press/v28/gittens13.pdf
45. Spectral Grouping Using the Nystro ̈m Method - https://people.eecs.berkeley.edu/~malik/papers/FBCM-nystrom.pdf
46. [LAURENS VAN DER MAATEN](https://lvdmaaten.github.io/)
47. http://lvdmaaten.github.io/drtoolbox/
48. Dimensionality Reduction: A Comparative Review. - http://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf
49. Visualizing Data using t-SNE - http://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf
50. Laplacian Eigenmaps for Dimensionality Reduction and Data Representation - https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf
51. Laplacian eigenmaps and spectral techniquesfor embedding and clustering - http://web.cse.ohio-state.edu/~belkin.8/papers/LEM_NIPS_01.pdf
52. Diffusion Maps: Analysis and Applications - https://core.ac.uk/download/pdf/1568327.pdf
53. Computing and Processing Correspondences with Functional Maps - http://www.lix.polytechnique.fr/~maks/fmaps_SIG17_course/notes/siggraph17_course_notes.pdf
54. Vector Diffusion Maps and the Connection Laplacian - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.435.8939&rep=rep1&type=pdf
55. Nonlinear Dimensionality Reduction II: Diffusion Maps - https://www.stat.cmu.edu/~cshalizi/350/lectures/15/lecture-15.pdf
56. Understanding the geometry of transport: diffusion maps for Lagrangian trajectory data unravel coherent sets - https://arxiv.org/pdf/1603.04709.pdf
57. Diffusion Maps, Spectral Clustering and Eigenfunctions of Fokker-Planck Operators - https://papers.nips.cc/paper/2942-diffusion-maps-spectral-clustering-and-eigenfunctions-of-fokker-planck-operators.pdf
58. Diffusion Maps for Signal Processing - http://www.eng.biu.ac.il/~gannot/articles/Diffusion%20Magazine.pdf
59. Applications of Diffusion Wavelets - https://core.ac.uk/reader/1145976
60. Diffusion Wavelets and Applications - http://helper.ipam.ucla.edu/publications/mgaws5/mgaws5_5164.pdf
61. Value Function Approximation with Diffusion Wavelets and Laplacian Eigenfunctions - https://people.cs.umass.edu/~mahadeva/papers/nips-paper1-v5.pdf
62. Wavelet methods in statistics: Some recent developments and their applications - https://arxiv.org/pdf/0712.0283.pdf
63. StatQuest: t-SNE, Clearly Explained - https://www.youtube.com/watch?v=NEaUSP4YerM
64. Shortest-path kernels on graphs - https://www.dbs.ifi.lmu.de/~borgward/papers/BorKri05.pdf
65. Generalized Shortest Path Kernel on Graphs - https://arxiv.org/pdf/1510.06492.pdf
66. Shortest-Path Graph Kernels for Document Similarity - https://www.aclweb.org/anthology/D17-1202.pdf
67. An Introduction to Graph Kernels - https://ethz.ch/content/dam/ethz/special-interest/bsse/borgwardt-lab/documents/slides/CA10_GraphKernels_intro.pdf
68. Fast shortest-path kernel computations using approximate methods - http://publications.lib.chalmers.se/records/fulltext/215958/215958.pdf
69. Shortest-path kernels on graphs - https://www.dbs.ifi.lmu.de/Publikationen/Papers/borgwardt.pdf
70. Graphlet Kernels - https://ethz.ch/content/dam/ethz/special-interest/bsse/borgwardt-lab/documents/slides/BNA09_3_4.pdf
71. Efficient graphlet kernels for large graph comparison - http://proceedings.mlr.press/v5/shervashidze09a/shervashidze09a.pdf
72. The Graphlet Spectrum - http://members.cbio.mines-paristech.fr/~nshervashidze/publications/KonSheBor09.pdf
73. Efficient graphlet kernels for large graph comparison - https://people.mpi-inf.mpg.de/~mehlhorn/ftp/AISTATS09.pdf
74. Generalized graphlet kernels for probabilistic inference in sparse graphs - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.720.557&rep=rep1&type=pdf
75. Graphlet Decomposition: Framework, Algorithms, and Applications - https://nickduffield.net/download/papers/KAIS-D-15-00611R2.pdf
76. Efficient Graphlet Counting for Large Networks - https://www.cs.purdue.edu/homes/neville/papers/ahmed-et-al-icdm2015.pdf
77. Halting in Random Walk Kernels - https://papers.nips.cc/paper/5688-halting-in-random-walk-kernels.pdf
78. Graph Kernels - http://www.jmlr.org/papers/volume11/vishwanathan10a/vishwanathan10a.pdf
79. Fast Random Walk Graph Kernel - http://www.cs.cmu.edu/~ukang/papers/fast_rwgk.pdf
80. GRAPH KERNELS - https://sites.cs.ucsb.edu/~xyan/tutorial/GraphKernels.pdf
81. Fast Computation of Graph Kernels - https://pdfs.semanticscholar.org/4459/336b270333c3666310a332acfb2641b27c0d.pdf
82. Weisfeiler-Lehman Graph Kernels - http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf
83. The Weisfeiler-Lehman Kernel - https://ethz.ch/content/dam/ethz/special-interest/bsse/borgwardt-lab/documents/slides/CA10_WeisfeilerLehman.pdf
84. Wasserstein Weisfeiler-Lehman Graph Kernels - https://papers.nips.cc/paper/8872-wasserstein-weisfeiler-lehman-graph-kernels.pdf
85. Global Weisfeiler-Lehman Kernels - https://arxiv.org/pdf/1703.02379.pdf
86. A Persistent Weisfeiler–Lehman Procedure for Graph Classification - http://proceedings.mlr.press/v97/rieck19a/rieck19a.pdf
87. A Fast Approximation of the Weisfeiler-Lehman Graph Kernel for RDF Data - https://work.delaat.net/awards/2013-09-23-paper.pdf
88. RDRToolbox A package for nonlinear dimension reduction with Isomap and LLE. - https://www.bioconductor.org/packages/release/bioc/vignettes/RDRToolbox/inst/doc/vignette.pdf
89. An Introduction to Diffusion Maps - https://inside.mines.edu/~whereman/talks/delaPorte-Herbst-Hereman-vanderWalt-DiffusionMaps-PRASA2008.pdf
90. Convergence of Laplacian Eigenmaps - http://papers.neurips.cc/paper/2989-convergence-of-laplacian-eigenmaps.pdf
91. Laplacian Eigenmap for Image Retrieval - http://people.cs.uchicago.edu/~niyogi/papersps/paper.pdf
92. Quantum Laplacian Eigenmap - https://arxiv.org/pdf/1611.00760.pdf
93. Laplacian Eigenmaps from Sparse, Noisy Similarity Measurements - https://arxiv.org/pdf/1603.03972.pdf
94. Laplacian eigenmaps for multimodal groupwise image registration - https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjIwqKVvqvqAhVU6qYKHeM7AA4QFjAKegQIChAB&url=https%3A%2F%2Frepub.eur.nl%2Fpub%2F100364%2FRepub_100364_O-A.pdf&usg=AOvVaw2xVygozmBAE735xOQ8m_rQ
95. Nonlinear Dimensionality Reduction I: Local Linear Embedding - https://www.stat.cmu.edu/~cshalizi/350/lectures/14/lecture-14.pdf
96. A NOTE ON THE LOCALLY LINEAR EMBEDDING ALGORITHM - https://core.ac.uk/reader/21747186
97. LOCALL Y LINEAR EMBEDDING ALGORI THM - http://jultika.oulu.fi/files/isbn9514280415.pdf
98. Truly Incremental Locally Linear Embedding - http://ai.stanford.edu/~schuon/learning/inclle.pdf
99. Supervised locally linear embedding - http://rduin.nl/papers/icann_03_lle.pdf
100. Me gusta en YouTube: On Graph Kernels - https://www.youtube.com/watch?v=xwVOarJGD7Q
101. Embedding & Manifold Learning - https://moodle.unive.it/mod/resource/view.php?id=176673
102. Wednesday 29/4/2020 - https://drive.google.com/file/d/1jQfGEqw9CYOHYAiIIdxmG7zc-7GKh5sY/view
103. Monday 27/4/2020 - https://drive.google.com/file/d/1IM9csbR7s-ec2_I_ck1GzOoZeaRAaFGx/view
104. Wednesday 22/4/2020 - https://drive.google.com/file/d/1wzkmQJ344orELbQKoVL1P-yrAMofi3v8/view
105. On Graph Kernels - https://www.youtube.com/watch?v=xwVOarJGD7Q
106. Weisfeiler-Lehman Neural Machine for Link Prediction - https://www.youtube.com/watch?v=QYhgLVt56z8
107. Deep Graph Kernels - https://www.youtube.com/watch?v=hqbMbTlTpXU
108. Graph Theory FAQs: 03. Isomorphism Using Adjacency Matrix - https://www.youtube.com/watch?v=UCle3Smvh1s
109. Deep Graph Kernels - https://dl.acm.org/doi/pdf/10.1145/2783258.2783417
110. GRAPHLET COUNTING - http://evlm.stuba.sk/APLIMAT2018/proceedings/Papers/0442_Hocevar.pdf
111. Graphlet based network analysis - https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2207&context=open_access_dissertations
112. Graphlet Counting for Topological Data Analysis - https://webthesis.biblio.polito.it/7641/1/tesi.pdf
113. Estimating Graphlet Statistics via Lifting - https://arxiv.org/pdf/1802.08736.pdf
114. GEM - https://github.com/palash1992/GEM
115. awesome-graph-classification - https://github.com/benedekrozemberczki/aweso/me-graph-classification
116. node2vec: Embeddings for Graph Data - https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef
117. Graph Embedding - Graph Analysis and Graph Learning - https://maelfabien.github.io/machinelearning/graph_5/#
118. DeepWalk: Implementing Graph Embeddings in Neo4j - https://neo4j.com/blog/deepwalk-implementing-graph-embeddings-in-neo4j/
119. Inference on Graphs with Support Vector Machines - http://members.cbio.mines-paristech.fr/~jvert/talks/040206insead/insead.pdf
120. sklearn.manifold.SpectralEmbedding - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold)
121. sklearn.manifold.LocallyLinearEmbedding - https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn-manifold-locallylinearembedding
122. Graph Classification - https://www.csc2.ncsu.edu/faculty/nfsamato/practical-graph-mining-with-R/slides/pdf/Classification.pdf
123. SVMS and kernel methods for graphs - https://courses.cs.ut.ee/2011/graphmining/Main/KernelMethodsForGraphs
124. Graph Representation Learning and Graph Classification - https://www.cs.uoregon.edu/Reports/AREA-201706-Riazi.pdf
