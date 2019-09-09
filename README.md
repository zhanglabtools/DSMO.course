# Data Science and Matrix Optimization

## About the Course
Data science is a "concept to unify statistics, data analysis, machine learning and their related methods" in order to "understand and analyze actual phenomena" with data<sup>[1](#myfootnote1)</sup>.  With the development of the technologies of data collection and storage, big data emerges from various fields.   It brings great opportunities for researchers. Many algorithms have been proposed , and most of them involve intensive matrix optimization techniques.  This course covers ten important topics of “Data Science” (one topic per week).  It is intended to teach mathematical models, matrix optimization models, algorithms and applications related to ten basic problems from practical problems and real-world data. This course is designed for doctoral, postgraduate and upper-level undergraduate students in all majors.

The ten topics and the corresponding material are as follows:
 1. **Robust PCA**  [material](#Robust-Principal-Component-Analysis) [slides](./course_files/lecture_slides/RobustPCA.pdf)
 2. **Non-negative Matrix Factorization** [material](#Nonnegative-Matrix-Factorization) [slides](./course_files/lecture_slides/NMF.pdf)
  3. **Matrix Completion** 
  4. **Sparse Coding**
  5. **Sparse Sensing**
  6. **Subspace Clustering**
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

##  Contact
If you have any comments, questions or suggestions about the material, please contact zhangchihao11@outlook.com

---
<a name="myfootnote1">1</a> . Hayashi, Chikio (1 January 1998). ["What is Data Science? Fundamental Concepts and a Heuristic Example"](https://www.springer.com/book/9784431702085)

