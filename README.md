# 常见 CUDA 算子实现

- [x] element-wise
- [x] transpose
- [x] reduction
- [ ] GEMV
- [ ] SGEMM
- [ ] convolution
- [ ] flash attention



## 1. element-wise

知乎文章：[CUDA element-wise 算子详解](https://zhuanlan.zhihu.com/p/1888630735520391519)

> GPU：NVDIA GeForce RTX 4060 Ti (Compute Capability 8.9)
> 
> CUDA: 12.8
> 
> N: 16 * 1024 * 1024

|              | <grid_size, block_size> | Memory Throughput (Gbytes/s) | Times(us) |
| ------------ | ----------------------- | ---------------------------- | --------- |
| 朴素实现     | <65536, 256>            | 256.02                       | 745.12    |
| 网格跨步循环 | <65536, 256>            | 259.08                       | 723.74    |
| 向量化访存   | <16384, 256>            | 259.23                       | 711.97    |

## 2. transpose

知乎文章：[CUDA transpose 算子详解](https://zhuanlan.zhihu.com/p/1899760505733756129)

> GPU：NVDIA GeForce RTX 4060 Ti (Compute Capability 8.9)
> 
> CUDA: 12.8
> 
> 矩阵尺寸：M = N = 9600

 |                               | block size | duration (ms) |
 | ----------------------------- | ---------- | ------------- |
 | NaiveRow                      | (4, 64)    | 2.75          |
 | NaiveRow                      | (8, 32)    | 2.87          |
 | NaiveCol                      | (4, 64)    | 2.81          |
 | NaiveCol                      | (8, 32)    | 2.85          |
 | ColNelements<16, 64>          | (4, 64)    | 3.00          |
 | ColNelements<32, 32>          | (8, 32)    | 2.99          |
 | Shared<32, 32>                | (32, 8)    | 3.01          |
 | SharedPadding<32, 32>         | (32, 8)    | 3.08          |
 | SharedSwizzling<32, 32>       | (32, 8)    | 3.13          |
 | SharedUnroll<32, 32>          | (32, 8)    | 3.05          |
 | SharedPaddingUnroll<32, 32>   | (32, 8)    | 3.02          |
 | SharedSwizzlingUnroll<32, 32> | (32, 8)    | 3.01          |
 | cuBLAS                        |            | 3.01          |


> GPU：NVIDIA GeForce GTX 960M (Compute Capability 5.0)
> 
> CUDA: 12.1
> 
> 矩阵尺寸：M = N = 9600

 |                               | block size | duration (ms) |
 | ----------------------------- | ---------- | ------------- |
 | NaiveRow                      | (8, 32)    | 21.57         |
 | NaiveCol                      | (8, 32)    | 16.31         |
 | ColNelements<16, 64>          | (4, 64)    | 11.20         |
 | ColNelements<32, 32>          | (8, 32)    | 15.78         |
 | Shared<32, 32>                | (32, 8)    | 22.27         |
 | SharedPadding<32, 32>         | (32, 8)    | 19.81         |
 | SharedSwizzling<32, 32>       | (32, 8)    | 19.65         |
 | SharedUnroll<32, 32>          | (32, 8)    | 18.87         |
 | SharedPaddingUnroll<32, 32>   | (32, 8)    | 7.73          |
 | SharedSwizzlingUnroll<32, 32> | (32, 8)    | 7.67          |
 | cuBLAS                        |            | 11.21         |


 ## 3. reduction

知乎文章：[CUDA reduce 算子详解](https://zhuanlan.zhihu.com/p/1905661893739283464)

> GPU：NVDIA GeForce RTX 4060 Ti (Compute Capability 8.9)
> 
> CUDA: 12.8
> 
> N: 16 * 1024 * 1024

|              | block size | elements per block | duration (us) | memeory throughput % |
| ------------ | ---------- | ------------------ | ------------- | -------------------- |
| reduce0      | 256        | 256                | 690.05        | 52.89                |
| reduce0.5    | 256        | 256                | 436.93        | 83.40                |
| reduce1      | 256        | 256                | 426.66        | 85.39                |
| reduce2      | 256        | 256                | 414.14        | 87.95                |
| reduce3      | 256        | 256 * 2            | 371.61        | 88.00                |
| reduce4      | 256        | 256 * 2            | 251.68        | 96.92                |
| reduce5      | 256        | 256 * 2            | 259.14        | 96.82                |
| reduce6      | 256        | 256 * 4            | 251.52        | 96.89                |
| reduce6_vec4 | 256        | 256 * 4            | 251.39        | 96.89                |
| reduce7      | 256        | 256 * 4            | 251.74        | 96.79                |