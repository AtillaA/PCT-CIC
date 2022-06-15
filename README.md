# Machine-Learning-for-3D-Geometry

# PCT: Point Cloud Transformer

This README will be updated soon with required definitions and citations regarding the original PCT implementation by Meng-Hao et al.

Paper link: https://arxiv.org/pdf/2012.09688.pdf


## <font color=red>News</font> :

* 2022.6.21 : Tasks carried out by date
* 2022.6.29 : ..


## Astract


Local feature aggregation is an operation that assembles point features of a given key point set, computes the position encodings of the subject point and the neighboring points, and passes the results into relevant transformation and aggregation modules in furtherance of local feature extraction. Even though these operations are feasible for depicting relative local patterns, they are inept with regard to long-range point relations. To that extent, the aggregation strategy introduced by Xiang et al.~\cite{xiang2021walk} proposes a new long-range feature aggregation method, namely curve aggregation, for point clouds shape analysis. Initiative of our project is to implement the curve aggregation method upon the Point Cloud Transformer (PCT) of Guo et al.~\cite{guo2021pct}, replacing the local neighbor embedding strategy.


## Dataset


Apart from the baseline datasets that are used in PCT (which are ModelNet40 for point cloud classification, ShapeNet for object segmentation, and Stanford 3D Indoor Dataset), we will be exploring various other point cloud datasets (e.g. Partnet) to evaluate and verify our findings.
