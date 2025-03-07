# 通用矩阵乘法 (General Matrix Multiplication, GEMM)

## 计算过程

### 1. 定义

   $C \leftarrow \alpha AB + \beta C$

   $A$ 的形状为 $M \times K$，$B$ 的形状为 $K \times N$，$C$ 的形状为 $M \times N$

### 2. 计算复杂度

   (1) 计算 $AB$ 的每个元素都有 $K$ 次乘法和 $K-1$ 次加法，共 $MN(2K-1)$ 次浮点运算

   (2) 计算 $\alpha AB$ 和 $\beta C$ 都需要 $MN$ 次乘法，共 $2MN$ 次浮点运算

   总浮点运算次数： $(2K+1)MN \approx 2KMN$ ，单位 FLOPS (Float Point Operations)

## 计算方案

### 1. naiveGEMM

   (1) 每个线程负责 $C$ 中一个元素计算，共需要 $M \times N$ 个线程

   (2) 设每个线程块维度为 $(B_m, B_n)$，如 $(32, 32)$。则网格数量为 $(\frac{M}{B_m}, \frac{N}{B_n})$

   访问次数：每个线程读取 $2K$ 个元素（$A$ 的一行，$B$ 的一列），共读取 $2MNK$ 次


