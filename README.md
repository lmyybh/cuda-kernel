# 常见 CUDA 算子实现

- [x] element-wise
- [ ] transpose
- [ ] reduce
- [ ] GEMV
- [ ] SGEMM
- [ ] convolution
- [ ] flash attention



## 1. element-wise

知乎文章：[CUDA element-wise 算子详解](https://zhuanlan.zhihu.com/p/1888630735520391519)

> GPU：NVDIA GeForce RTX 4060 Ti
> 
> CUDA: 12.8
> 
> N: 16 * 1024 * 1024

|              | <grid_size, block_size> | Memory Throughput (Gbytes/s) | Times(us) |
| ------------ | ----------------------- | ---------------------------- | --------- |
| 朴素实现     | <65536, 256>            | 256.02                       | 745.12    |
| 网格跨步循环 | <65536, 256>            | 259.08                       | 723.74    |
| 向量化访存   | <16384, 256>            | 259.23                       | 711.97    |