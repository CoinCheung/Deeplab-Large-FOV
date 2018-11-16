baseline: 
1. 对score map做插值而不是对label做插值
2. xavier或msra初始化
4. warmup 1000 iter
62.15

=====
1. 使用没aug的trainval

1. 使用aug的数据集
2. 使用crf
3. mixup
5. 加BN
