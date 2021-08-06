#求一个序列的概率

import numpy as np

def forward(psi):
    m, V, _ = psi.shape

    alpha = np.zeros([m, V])
    alpha[0] = psi[0, 0, :]  # assume psi[0, 0, :] := psi(*,s,1)

    for t in range(1, m):
        for i in range(V):
            '''
            for k in range(V):
                alpha[t, i] += alpha[t - 1, k] * psi[t, k, i]
            '''
            alpha[t, i] = np.sum(alpha[t - 1, :] * psi[t, :, i])

    return alpha

def pro(seq, psi):
    m, V, _ = psi.shape
    alpha = forward(psi)

    Z = np.sum(alpha[-1])
    M = psi[0, 0, seq[0]]
    for i in range(1, m):
        M *= psi[i, seq[i-1], seq[i]]

    p = M / Z
    return p

np.random.seed(1111)
V, m = 2, 4

# log_psi = np.random.random([m, V, V])
# print(log_psi)
log_psi = np.array([[
            [.5, .5],    #y0->y1
            [.0, .0]
        ],
        [
            [.7, .3],    #y1->y2
            [.4, .6]

        ],
        [
            [.2, .8],    #y2->y3
            [.5, .5]
        ],
        [
            [.9, .1],   #y3->y4
            [.8, .2]
        ]])
#print(log_psi.shape)

psi = np.exp(log_psi)  # nonnegative
#seq = np.random.choice(V, m)
seq = [1, 1, 1, 1]
#y1 = y2 = y3 = y4 =1时的概率
#print(seq)
alpha = forward(psi)
p = pro(seq, psi)
print("序列为："+repr(seq)+"的概率为："+repr(p))
#print(p)
#print(alpha)