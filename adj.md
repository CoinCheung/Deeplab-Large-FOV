baseline: 
1. 对score map做插值而不是对label做插值
2. xavier或msra初始化
4. warmup 1000 iter
val - 62.15, 62.05

=====
1. 使用没aug的trainval
test - 没有测试集gt

5. mixup 
目测不好使
alpha=1, 5e-4: 55.73
alpha=0.1, 5e-4: 60.56
alpha=0.1, 1e-4: 61.28
alpha=0.1, 1e-4, [4000, 6000]: 62.831， 高了一丢丢
alpha=0.1, 1e-4, [5000, 7000]: 

2. 使用aug的数据集
3. 使用crf
4. 使用Msc
