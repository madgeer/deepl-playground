"""Logistic regression from scratch — sigmoid, BCE, softmax.
Week 3 lab, Deep Learning."""
import numpy as np

def sigmoid(z):
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

def bce_loss(p, y, eps=1e-15):
    """Mean binary cross-entropy.
    The 1/N is what makes eta portable across batch sizes, and what
    makes
    the initial loss equal log 2 rather than N log 2."""
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def softmax_naive(Z):
    """Softmax exactly as written on the slide. Overflows for large
    logits.
    Used deliberately in B.4 so that the failure is discovered, not
    described."""
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def softmax_stable(Z):
    """Same function, shifted by the row max: softmax(z + c) =
    softmax(z)."""
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

def ce_loss(P, Y_idx, eps=1e-15):
    """Mean categorical cross-entropy. Y_idx holds class indices,
    not one-hot."""
    P = np.clip(P, eps, 1 - eps)
    return -np.mean(np.log(P[np.arange(len(Y_idx)), Y_idx]))

def grads_binary(X, y, w, b):
    """dL/dw and dL/db for sigmoid + mean BCE.
    Convention: rows are examples (see A.9)."""
    p = sigmoid(X @ w + b)
    d = p - y
    return X.T @ d / len(y), d.mean()

def finite_diff(f, theta, h=1e-5):
    """Central difference, elementwise. Mutates theta in place and
    restores it."""
    g = np.zeros_like(theta, dtype=np.float64)
    it = np.nditer(theta, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        old = theta[i]
        theta[i] = old + h
        a = f()
        theta[i] = old - h
        c = f()
        theta[i] = old
        g[i] = (a - c) / (2 * h)
        it.iternext()
    return g
