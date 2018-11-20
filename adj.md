baseline: 
1. 对score map做插值而不是对label做插值
2. xavier或msra初始化
4. warmup 1000 iter
val - 62.15, 62.05

=====
1. 使用没aug的trainval
test - 没有测试集gt

2. mixup 
alpha=1, 5e-4: 55.73
alpha=0.1, 5e-4: 60.56
alpha=0.1, 1e-4: 61.28
alpha=0.1, 1e-4, [4000, 6000]: 62.831， 高了一丢丢
alpha=0.1, 1e-4, [5000, 7000]: 63.10, 高了不少呢
没有mixup, 5e-5, [5000, 7000]: 63.12, 看样子一样的

3. 去掉dropout
63.20, 跟原来差不多

4. 使用aug的数据集
去掉dropout的: 67.77

3. 使用crf
4. 使用Msc
