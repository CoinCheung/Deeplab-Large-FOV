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
没有mixup, 5e-5, [5000, 7000]:  63.12, 63.46, 看样子一样的
没有mixup, 5e-5, [4000, 5500]:  62.95, 

3. 去掉dropout
没有mixup
63.20, 跟原来差不多
[4000, 5500]/6000: 63.085，没啥差别的意思
加上mixup:

4. 使用aug的数据集
1) 去掉dropout的: 
没有mixup, 5e-5
67.77, 67.94 - 提高不少

加上mixup:
alpha=0.1, 1e-4, [5000, 7000]/8000: 66.94，看样子mixup无用 
alpha=0.1, 1e-4, [10000, 14000]/16000: 69.19，

不加mixup:
[10000, 14000]/16000, 1e-4: 69.67(no_crf)，70.75(crf)还是比mixup好
[10000, 14000]/16000, 5e-4: 69,63(no_crf), 70.76


5. batch size
crop size: 321
batchsize = 20, [12500, 17500]/20000: 70.16
batchsize = 30, [12500, 17500]/20000: 70.32
看样子影响不大，就好一丢丢
crop size: 497
batchsize=30, [12500, 17500]/20000: 67.58 
batchsize=20, [12500, 17500]/20000: 67.51
alpha=0.1, 1e-4, batchsize=20, crop=497, [12500, 17500]/20000: 66.83


2) 加上dropout的: 
[10000, 14000]/16000, 5e-4: 


6. crop size
改成513: 
memory 限制batchsize = 20, [10000, 14000]/16000, (no_crf): 66.74
batchsize = 20, [12500, 17500]/20000, (no_crf): 67.18



7. multi-scale training和testing
batchsize=20, (457, 457): screen-40167
batchsize=20, (497, 497), 5e-4: 71.82，有用了
batchsize=20, (497, 497), 5e-4, dropout:  71.84(no_crf), 72.23(crf)


8. 使用crf
voc2012: (3,3,4,121,5), 5: 63.72，一丢丢
voc2012: (3,3,4,121,5), 5: 64.51，强了不少呢
