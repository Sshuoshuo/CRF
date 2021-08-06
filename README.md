# -CRF
CRF三个任务的实现
compute_pro.py是解决概率计算问题，计算一个序列的概率，使用前向算法
forward_backward.py是计算P(yi|x)，使用的是前向-后向算法
learning.py是CRF的参数学习问题，运用了sklearn_crfsuite的API，以conll2002数据集进行训练
decoding.py是CRF的解码/预测问题，运用维特比算法求解了李航书P234例11.3
