# 通用矩阵乘法 (General Matrix Multiplication, GEMM)

## 计算过程

1. 定义

   $C \leftarrow \alpha AB + \beta C$

   $A$ 的形状为 $M \times K$，$B$ 的形状为 $K \times N$，$C$ 的形状为 $M \times N$

2. 计算复杂度

   (1) 计算 $AB$ 的每个元素都有 $K$ 次乘法和 $K-1$ 次加法，共 $MN(2K-1)$ 次浮点运算
   (2) 计算 $\alpha AB$ 和 $\beta C$ 都需要 $MN$ 次乘法，共 $2MN$ 次浮点运算

   总浮点运算次数：$(2K+1)MN \approx 2KMN$，单位 FLOPS (Float Point Operations)
