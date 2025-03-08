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

#### 1.1 方法

   (1) 每个线程读取 $A$ 中一行， $B$ 中一列，负责 $C$ 中一个元素计算，共需要 $M \times N$ 个线程

   (2) 设每个线程块维度为 $(B_m, B_n)$，如 $(32, 32)$。则 `grid` 维度为 $(\frac{M}{B_m}, \frac{N}{B_n})$

#### 1.2 性能分析

**(1) `global memory` 读取次数**
   
   每个线程读取 $2K$ 个元素（ $A$ 的一行， $B$ 的一列），共读取 $2MNK$ 次

### 2. blockGEMM

#### 2.1 方法

   (1) 对 $C$ 切块，每个 `block` 负责 $C$ 中 $B_m \times B_n$ 大小切块的计算。则 `grid` 维度为 $(\frac{M}{B_m}, \frac{N}{B_n})$

   (2) 对于每个 `block`，需要读取 $A$ 中 $B_m \times K$ 个元素， $B$ 中 $K \times B_n$ 个元素，放到 `shared memory`。由于 $K$ 可能很大，导致 $A$ 和 $B$ 的切块无法放入 `shared memory`，因此需要进一步划分。

   <div  align="center"><img src="./assets/01_blockGEMM.png" width=300></img></div>

   (3) 对于每个 `block` 负责的切块，在 $K$ 维度上进一步划分为 $B_k$ 大小的块，共划分为 $\frac{K}{B_k}$ 个块，即每个 `block` 循环计算 $\frac{K}{B_k}$ 次。在每次循环中，读取 $A$ 中 $B_m \times B_k$ 个元素， $B$ 中 $B_k \times B_n$ 个元素，放到 `shared memory` 中。

   <div  align="center"><img src="./assets/02_blockGEMM_Bk.png" width=300></img></div>

   (4) 同理，对于每个线程也可以进行切块，分别将 $B_m$, $B_n$, $B_k$ 三个维度切分为 $T_m$, $T_n$, $1$ 大小的块，即每个 `thread` 循环计算 $B_k$ 次。在每次循环中，读取 $A$ 中 $T_m$ 个元素， $B$ 中 $T_n$ 个元素，放到 `register` 中。

   <div  align="center"><img src="./assets/03_blockGEMM_Tk.png" width=300></img></div>

   (5)  `grid` 维度为 $(\frac{M}{B_m}, \frac{N}{B_n})$ ，`block` 维度为 $(\frac{B_m}{T_m}, \frac{B_n}{T_n})$

#### 2.2 性能分析

**(1) `global memory` 读取次数**

   每个 `block` 中的每次循环需要读取 $A$ 中 $(B_m, B_k)$ 大小的切块和 $B$ 中 $(B_k, B_n)$ 大小的切块，共循环 $\frac{K}{B_k}$ 次，则每个 `block` 共读取 $K(B_m + B_n)$ 次。而 `grid` 维度为 $(\frac{M}{B_m}, \frac{N}{B_n})$ ，因此，总的 `global memory` 读取次数为 
   
   $$\frac{M}{B_m} \times \frac{N}{B_n} \times K(B_m + B_n) = MNK(\frac{1}{B_m} + \frac{1}{B_n})$$


# 对比

**测试环境**

> GPU: NVIDIA GeForce RTX 4060 Ti 
> 
> Version: Cuda12.8

| 方法      | Global Memory 读取次数               | 运行时间 | 缺陷                             |
| --------- | ------------------------------------ | -------- | -------------------------------- |
| naiveGEMM | $2MNK$                               | 1.575 ms | global memory 访问次数多，带宽低 |
| blockGEMM | $MNK(\frac{1}{B_m} + \frac{1}{B_n})$ | 0.858 ms | 存在 bank conflict               |



# 参考资料

[1] [猛猿，“从啥也不会到 CUDA GEMM 优化”，知乎，2024](https://zhuanlan.zhihu.com/p/703256080)

[2] [有了琦琦的棍子，“深入浅出GPU优化系列：GEMM优化 (一) (二) (三)”，知乎，2022](https://zhuanlan.zhihu.com/p/435908830)
