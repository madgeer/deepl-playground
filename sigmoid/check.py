import numpy as np
from logistic import *

rng = np.random.default_rng(3)
N, d, K = 40, 6, 10
X = rng.normal(size=(N, d))
y = (rng.random(N) < 0.5).astype(np.float64)
yi = rng.integers(0, K, N)

# log K check — independent of data and seed
w, b = np.zeros(d), 0.0
print("zero-init BCE = %.9f (log 2 = %.9f)"
% (bce_loss(sigmoid(X @ w + b), y), np.log(2)))
W, bb = np.zeros((d, K)), np.zeros(K)
print("zero-init CE (K=10) = %.9f (log 10 = %.9f)"
% (ce_loss(softmax_stable(X @ W + bb), yi), np.log(10)))

# gradient check
w = rng.normal(size=d) * 0.5
b = 0.3
ga, _ = grads_binary(X, y, w, b)
gn = finite_diff(lambda: bce_loss(sigmoid(X @ w + b), y), w)
print("max |analytic - numeric| = %.3e" % np.abs(ga - gn).max())

# overflow demonstration
Z = np.array([[800.0, 799.0, 1.0]])
print("naive :", softmax_naive(Z))
print("stable:", softmax_stable(Z))