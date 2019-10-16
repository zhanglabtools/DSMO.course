# Data Science and Matrix Optimization

## About the Course
Data science is a "concept to unify statistics, data analysis, machine learning and their related methods" in order to "understand and analyze actual phenomena" with data<sup>[1](#myfootnote1)</sup>.  With the development of the technologies of data collection and storage, big data emerges from various fields.   It brings great opportunities for researchers. Many algorithms have been proposed , and most of them involve intensive matrix optimization techniques.  This course covers ten important topics of “Data Science” (one topic per week).  It is intended to teach mathematical models, matrix optimization models, algorithms and applications related to ten basic problems from practical problems and real-world data. This course is designed for doctoral, postgraduate and upper-level undergraduate students in all majors.

The ten topics and the corresponding material are as follows:
 1. **Robust PCA**  [material](#Robust-Principal-Component-Analysis) [slides](./course_files/lecture_slides/RobustPCA.pdf)
 2. **Non-negative Matrix Factorization** [material](#Nonnegative-Matrix-Factorization) [slides](./course_files/lecture_slides/NMF.pdf)
  3. **Matrix Completion** [material](#Matrix-Completion) [slides](./course_files/lecture_slides/MatrixCompletion.pdf)
  4. **Sparse Coding** [material](#Sparse-Coding) [slides](./course_files/lecture_slides/SparseCoding.pdf)
  5. **Sparse Sensing** [material](#Sparse-Sensing) [slides](./course_files/lecture_slides/CompressedSensing.pdf)
  6. **Subspace Clustering** [material](#Subspace-Clustering) [slides](./course_files/lecture_slides/SubspaceClustering.pdf)
  7. **Precision Matrix Estimation**
  8. **Nonlinear Manifold Learning**
  9. **Manifold Alignment**
  10. **Tensor Factorization**

##  Prerequisites

Mathematical Analysis, Linear Algebra

**Optional**:  Mathematical Statistics , Numerical Optimization, Matrix Theory

## Robust Principal Component Analysis

### Software
+ The [**LRSLibrary**](https://github.com/andrewssobral/lrslibrary) provides a collection of low-rank and sparse decomposition algorithms in MATLAB. In the [RPCA](https://github.com/andrewssobral/lrslibrary/tree/master/algorithms/rpca) section, The MATLAB codes of Accelerated Proximal Gradient Method (APGM), the Exact Augmented Lagrange Multiplier(EALM) and the Inexact Augmented Lagrange Multiplier(IALM) can be available.
+ The MATLAB code of the Alternating Splitting Augmented Lagrangian Method(ASALM) can be obtained [here](https://github.com/andrewssobral/lrslibrary/blob/947cd52bc27c616cab7a7a668d55b51e635553d4/algorithms/rpca/SPGL1/recreatePaperExperiment.m).

+ [ADMIP](http://www2.ie.psu.edu/aybat/codes.html): Alternating Direction Method with Increasing Penalty(MATLAB code)

+ The MATLAB code of Low-rank Matrix Fitting[(LMafit)](http://lmafit.blogs.rice.edu/)

### Key papers
+ Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust principal component analysis?. Journal of the ACM (JACM), 58(3), 11.
+ Ma, S., & Aybat, N. S. (2018). Efficient optimization algorithms for robust principal component analysis and its variants. Proceedings of the IEEE, 106(8), 1411-1426.
+ Wright, J., Ganesh, A., Rao, S., Peng, Y., & Ma, Y. (2009). Robust principal component analysis: Exact recovery of corrupted low-rank matrices via convex optimization. In Advances in neural information processing systems (pp. 2080-2088).
+ Lin, Z., Ganesh, A., Wright, J., Wu, L., Chen, M., & Ma, Y. (2009). Fast convex optimization algorithms for exact recovery of a corrupted low-rank matrix. Coordinated Science Laboratory Report no. UILU-ENG-09-2214, DC-246.
+ Lin, Z., Chen, M., & Ma, Y. (2010). The augmented lagrange multiplier method for exact recovery of corrupted low-rank matrices. arXiv preprint arXiv:1009.5055.
+ Zhou, Z., Li, X., Wright, J., Candes, E., & Ma, Y. (2010, June). Stable principal component pursuit. In 2010 IEEE international symposium on information theory (pp. 1518-1522). IEEE.
+ Tao, M., & Yuan, X. (2011). Recovering low-rank and sparse components of matrices from incomplete and noisy observations. SIAM Journal 
on Optimization, 21(1), 57-81.
+ Aybat, N. S., & Iyengar, G. (2015). An alternating direction method with increasing penalty for stable principal component pursuit. Computational Optimization and Applications, 61(3), 635-668.
+ Lin, T., Ma, S., & Zhang, S. (2018). Global convergence of unmodified 3-block ADMM for a class of convex minimization problems. Journal of Scientific Computing, 76(1), 69-88.
+ Shen, Y., Wen, Z., & Zhang, Y. (2014). Augmented Lagrangian alternating direction method for matrix separation based on low-rank factorization. Optimization Methods and Software, 29(2), 239-263.



##  Nonnegative Matrix Factorization

### Software 

- MATLAB have a built-in function `nnmf`
- [Nimfa](http://nimfa.biolab.si/):  a Python library for nonnegative matrix factorization. It includes implementations of several factorization methods, initialization approaches, and quality scoring. Both dense and sparse matrix representation are supported. 
- Graph Regularized NMF ([MATLAB code](http://www.cad.zju.edu.cn/home/dengcai/Data/code/GNMF.m))
- JMF:   (**J**oint **M**atrix **F**actorization) is a MATLAB package to integrate multi-view data as well as prior relationship knowledge within or between multi-view data for pattern recognition and data mining. (MATLAB code available at [here](http://page.amss.ac.cn/shihua.zhang/software.html)) 
- CSMF:  (**C**ommon and **S**pecific **M**atrix **F**actorization) is a MATLAB package to simultaneously simultaneously extract common and specific patterns from the data of two or multiple biological interrelated conditions via matrix factorization. (MATLAB code available at [here](http://page.amss.ac.cn/shihua.zhang/software.html)) 

###  Key papers

- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, *401*(6755), 788.
- Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In *Advances in neural information processing systems* (pp. 556-562).
- Feng, T., Li, S. Z., Shum, H. Y., & Zhang, H. (2002, June). Local non-negative matrix factorization as a visual representation. In *Proceedings 2nd International Conference on Development and Learning. ICDL 2002* (pp. 178-183). IEEE.
- Hoyer, P. O. (2004). Non-negative matrix factorization with sparseness constraints. *Journal of machine learning research*, *5*(Nov), 1457-1469.
- Ding, C. H., Li, T., & Jordan, M. I. (2008). Convex and semi-nonnegative matrix factorizations. *IEEE transactions on pattern analysis and machine intelligence*, *32*(1), 45-55.
- Kim, H., & Park, H. (2008). Nonnegative matrix factorization based on alternating nonnegativity constrained least squares and active set method. *SIAM journal on matrix analysis and applications*, *30*(2), 713-730.
- Vavasis, S. A. (2009). On the complexity of nonnegative matrix factorization. *SIAM Journal on Optimization*, *20*(3), 1364-1377.
- Cai, D., He, X., Han, J., & Huang, T. S. (2010). Graph regularized nonnegative matrix factorization for data representation. *IEEE transactions on pattern analysis and machine intelligence*, *33*(8), 1548-1560.
- Wang, Y. X., & Zhang, Y. J. (2012). Nonnegative matrix factorization: A comprehensive review. *IEEE Transactions on Knowledge and Data Engineering*, *25*(6), 1336-1353.
- Guan, N., Tao, D., Luo, Z., & Yuan, B. (2012). NeNMF: An optimal gradient method for nonnegative matrix factorization. *IEEE Transactions on Signal Processing*, *60*(6), 2882-2898.

##  Matrix Completion

### Software 
- [*SVT*](http://svt.stanford.edu/code.html) is a library written with matlib by [Emmanuel Candès](http://statweb.stanford.edu/~candes/) and Stephen Becker for *Exact Matrix Completion*. The algorithm is described in the paper [A singular value thresholding algorithm for matrix completion](http://statweb.stanford.edu/~candes/papers/SVT.pdf).

- [*Soft-Impute*](stat.columbia.edu/~rahulm/software.html) is a library for Approximate nuclear norm minimization written with matlib and R.

- [*PMF library*](http://www.cs.toronto.edu/~rsalakhu/BPMF.html) is a library for probabilistic matrix factorization by Ruslan Salakhutdinov with matlab. The objective is also the most common optimization objective in matrix factorization.

- [*GCMC*](https://github.com/RyanLu32/GCMC) is python library for [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263).

###  Key papers
- Candes,E.J and Recht,B. (2011). Exact matrix completion via convex optimization. *Foundations of Computational mathematics*, *9*(6), 717.
- Cai, Jian-Feng and Candes, Emmanuel J and Shen, Zuowei. (2010). A singular value thresholding algorithm for matrix completion. *SIAM Journal on Optimization*, *20*(4), 1956–1982.
- Mazumder, R., Hastie, T. J., and Tibshirani, R. (2010). Spectral regularization algorithms for learning large incomplete matrices.
*Journal of machine learning research : JMLR*, *11*, 2287–2322.
- SALAKHUTDINOV, R. (2008). Probabilistic matrix factorization. *Advances in Neural Information Processing Systems*, *20*, 1257–1264.
- Zhou, Y., Wilkinson, D. M., Schreiber, R., and Rong, P. (2008). Large-scale parallel collaborative filtering for the netflix prize.
*In Proc Intl Conf Algorithmic Aspects in Information Management*.
- Kalofolias, V., Bresson, X., Bronstein, M., and Vandergheynst, P. (2014). Matrix completion on graphs. *Computer Science*.
- Gemulla, R., Nijkamp, E., Haas, P. J., and Sismanis, Y. (2011). Large-scale matrix factorization with distributed stochastic gradient descent. *In Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 69–77.
- Rao, N., Yu, H.-F., Ravikumar, P., and Dhillon, I. S. (2015). Collaborative filtering with graph information: Consistency and scalable methods. *In Proceedings of the 28th International Conference on Neural Information Processing Systems*, *2*(15), 2107–2115.
- Sun, D. L. and Fevotte, C. (2014). Alternating direction method of multipliers for non-negative matrix factorization with the beta-divergence. *In IEEE International Conference on Acoustics*.
- Berg, Rianne van den and Kipf, Thomas N and Welling, Max. (2017). Graph convolutional matrix completion. *arXiv preprint* arXiv:1706.02263.

## Sparse Coding

### Software
- [*KSVD-Box v13*](http://www.cs.technion.ac.il/~ronrubin/Software/ksvdbox13.zip) : Implementation of the K-SVD and Approximate K-SVD dictionary training algorithms, and the K-SVD Denoising algorithm.
-	[*OMP-Box v10*](http://www.cs.technion.ac.il/~ronrubin/Software/ompbox10.zip) : Implementation of the Batch-OMP and OMP-Cholesky algorithms for quick sparse-coding of large sets of signals.
-	[*SparseLab*](http://www.cs.technion.ac.il/~ronrubin/Software/ompbox10.zip) is a library of Matlab routines for finding sparse solutions to underdetermined systems. 
-	You can get more information about such software on [*Elad’s homepage*](http://www.cs.technion.ac.il/~ronrubin/Software/ompbox10.zip).

###  Key papers
- Olshausen, B. and Field, D. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381:607–609.
- Aharon, M., Elad, M., and Bruckstein, A. (2006). K-svd: An algorithm for designing overcomplete dictionaries for sparse representation. *IEEE Transactions on signal processing*, *54*(11):4311–4322.
- Daubechies, I., Defrise, M., and De Mol, C. (2004). An iterative thresholding algorithm for linear inverse problems with a sparsity constraint. *Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences*, *57*(11):1413–1457.
- Li, Y. and Osher, S. (2009). Coordinate descent optimization for l1 minimization with application to compressed sensing; a greedy algorithm. *Inverse Problems and Imaging*, *3*(3):487–503.
- Gregor, K. and LeCun, Y. (2010). Learning fast approximations of sparse coding. *In Proceedings of the 27th International Conference on International Conference on Machine Learning*, pages 399–406. Omnipress.
- Papyan, V., Romano, Y., Sulam, J., and Elad, M. (2017). Convolutional dictionary learning via local processing. *In Proceedings of the IEEE International Conference on Computer Vision*, pages 5296–5304.
- Sulam, J., Papyan, V., Romano, Y., and Elad, M. (2018). Multilayer convolutional sparse modeling: Pursuit and dictionary learning.
*IEEE Transactions on Signal Processing*, *66*(15):4090–4104.

## Sparse Sensing

### Software
- [*SparseLab*](http://sparselab.stanford.edu/) is a library of Matlab routines for finding sparse solutions to underdetermined systems. It not only aims to provide tools for sparse representation in a cohesive package to the research community, if also allows researchers in this area to publicly release the code accompanying their published papers.
- [GPSR](http://www.lx.it.pt/~mtf/GPSR/)：Gradient Projection  for Sparse  Reconstruction a solver of gradient projection type, using special line search and termination techniques, gave faster solutions on our test problems than other techniques that had been proposed previously, including interior-point techniques.  A debiasing step based on the conjugate-gradient algorithm improves the results further.
- [*MPTK*](http://mptk.irisa.fr/) The Matching Pursuit Tool Kit (MPTK) provides a fast implementation of the Matching Pursuit algorithm for the sparse decomposition of multichannel signals.
- Reproducible Deep Compressive Sensing Collection of source code for deep learning-based compressive sensing (DCS) can be found [*here*](https://github.com/ngcthuong/Reproducible-Deep-Compressive-Sensing).

### Key papers
- Candes, E. and Tao, T. (2004). Near optimal signal recovery from random projections: Universal encoding strategies? *arXiv preprint math*/0410542.
- Candes, E. and Tao, T. (2005). Decoding by linear programming. *arXiv preprint math*/0502327.
- Candes, E. J., Romberg, J. K., and Tao, T. (2006). Stable signal recovery from incomplete and inaccurate measurements. *Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences*, *59*(8):1207–1223.
- Chen, S. S., Donoho, D. L., and Saunders, M. A. (2001). Atomic decomposition by basis pursuit. *SIAM review*, *43*(1):129–159.
- Donoho, D. L. and Elad, M. (2003). Optimally sparse representation in general (nonorthogonal) dictionaries via l1 minimization. *Proceedings of the National Academy of Sciences*, *100*(5):2197–2202.
- Duarte, M. F., Davenport, M. A., Takhar, D., Laska, J. N., Sun, T., Kelly, K. F., and Baraniuk, R. G. (2008). Single-pixel imaging via compressive sampling. *IEEE signal processing magazine*, *25*(2):83–91.
- Ji, S., Xue, Y., Carin, L., et al. (2008). Bayesian compressive sensing. *IEEE Transactions on signal processing*, *56*(6):2346
- Tropp, J. A. (2004). Greed is good: Algorithmic results for sparse approximation. *IEEE Transactions on Information theory*, *50*(10):2231–2242.
- Wu, Y., Rosca, M., and Lillicrap, T. P. (2019). Deep compressed sensing. *CoRR*, abs/1905.06723
- Xu, Z., Zhang, H., Wang, Y., Chang, X., and Liang, Y. (2010). L 1/2 regularization. *Science China Information Sciences*, *53*(6):1159–1169.

##  Subspace Clustering

### Software 

- [SPARSE SUBSPACE CLUSTERING](http://vision.jhu.edu/code/) Sparce Subspace Clustering (SSC) is a sparse representation and spectral clustering based method for clustering a set of data points lying in a union of low-dimensional subspaces into their respective subspaces. Please visit the corresponding SSC research page for more information. Depending on how sparse representation is computed, there are two variances of SSC.
- [LOW-RANK SUBSPACE CLUSTERING (LRSC)](http://vision.jhu.edu/code/) The code below is the low-rank subspace clustering code used in our experiments for our CVPR 2011 publication. We note that if your objective is subspace clustering, then you will also need some clustering algorithm. We found that spectral clustering from Ng, Jordan et. al. performed the best.
- [LOW-RANK-REPRESENTATION](https://sites.google.com/site/guangcanliu/) is the code for low rank representation.
4.	[DEEP-SUBSPACE-CLUSTERING](https://github.com/panji1990/Deep-subspace-clustering-networks) is a tensorflow implementation for our NIPS'17 paper: Pan Ji*, Tong Zhang*, Hongdong Li, Mathieu Salzmann, Ian Reid. Deep Subspace Clustering Networks. in NIPS'17.


###  Key papers

- Ehsan Elhamifar; René Vidal(2013) Sparse Subspace Clustering: Algorithm, Theory, and Applications. *IEEE Transactions on Pattern Analysis and Machine Intelligence* ( Volume: 35 , Issue: 11 , Nov. 2013 )
-	Canyi Lu; Jiashi Feng; Zhouchen Lin; Tao Mei; Shuicheng Yan(2019). Subspace Clustering by Block Diagonal Representation.
*IEEE Transactions on Pattern Analysis and Machine Intelligence* ( Volume: 41 , Issue: 2 , Feb. 1 2019 )
-	Mahdi Soltanolkotabi, Ehsan Elhamifar and Emmanuel J. Candes(2013) Robust subspace clustering. *The Annals of Statistics*. *42*(2) · January 2013
-	Guangcan Liu, Zhouchen Lin and Yong Yu(2010) Robust Subspace Segmentation by Low-Rank Representation. *Proceedings of the 27th International Conference on Machine Learning*, Haifa, Israel, 2010.
-	Can-Yi Lu, Hai Min, Zhong-Qiu Zhao, Lin Zhu, De-Shuang Huang, and Shuicheng Yan(2012) Robust and Efficient Subspace Segmentation via Least Squares Regression. *ECCV* 2012
-	Pan Ji, Tong Zhang, Hongdong Li, Mathieu Salzmann and Ian Reid(2017) Deep Subspace Clustering Networks.
*31st Conference on Neural Information Processing Systems (NIPS 2017)*, Long Beach, CA, USA.
-	Ulrike von Luxburg (2007) A Tutorial on Spectral Clustering.



##  Contact
If you have any comments, questions or suggestions about the material, please contact zhangchihao11@outlook.com

---
<a name="myfootnote1">1</a> . Hayashi, Chikio (1 January 1998). ["What is Data Science? Fundamental Concepts and a Heuristic Example"](https://www.springer.com/book/9784431702085)

