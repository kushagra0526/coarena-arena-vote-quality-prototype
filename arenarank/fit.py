"""Rating methods on the vote list.

naive    Bradley-Terry penalized ML, every vote weight 1, ties as half a win (what a plain leaderboard refit does)
elo      streaming Elo, K=16, votes in arrival order
capped   Bradley-Terry with each voter's total weight capped at `cap` votes
robust   Bradley-Terry + a left-side term, with a per-voter honesty weight from EM: p(honest | votes) against a coin-flip model,
         voter weight = honesty posterior, total weight per voter capped
"""
import numpy as np
from scipy.optimize import minimize
from .sim import sigmoid


def bt(d, w, lam=0.05, left_term=False):
    M = d["n_models"]
    L, R, y = d["left"], d["right"], d["y"]

    def f(p):
        th, g = p[:M], (p[M] if left_term else 0.0)
        s = th[L] - th[R] + g
        pr = sigmoid(s)
        ll = w * (y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12))
        res = w * (y - pr)
        grad = np.zeros(M + 1)
        np.add.at(grad, L, res); np.add.at(grad, R, -res)
        if left_term:
            grad[M] = res.sum()
        grad[:M] -= lam * th
        return -(ll.sum() - 0.5 * lam * (th ** 2).sum()), -grad
    p = minimize(f, np.zeros(M + 1), jac=True, method="L-BFGS-B").x
    th = p[:M] - p[:M].mean()
    return th, p[M]


def naive(d):
    return bt(d, np.ones(len(d["y"])))[0]


def elo(d, lr=0.03):
    th = np.zeros(d["n_models"])
    for l, r, y in zip(d["left"], d["right"], d["y"]):
        e = sigmoid(th[l] - th[r])
        th[l] += lr * (y - e); th[r] -= lr * (y - e)
    return th - th.mean()


def capped_weights(d, w, cap):
    tot = np.bincount(d["voter"], weights=w, minlength=d["n_voters"])
    scale = np.minimum(1.0, cap / np.maximum(tot, 1e-9))
    return w * scale[d["voter"]]


def capped(d, cap=15):
    return bt(d, capped_weights(d, np.ones(len(d["y"])), cap))[0]


def robust(d, cap=15, iters=8, prior=0.8):
    n = len(d["y"])
    h = np.full(d["n_voters"], prior)
    for _ in range(iters):
        w = capped_weights(d, h[d["voter"]], cap)
        th, g = bt(d, w, left_term=True)
        pr = sigmoid(th[d["left"]] - th[d["right"]] + g)
        # honest likelihood of each vote vs a coin flip (ties count as half a vote each way)
        lh = np.where(d["y"] == 0.5, 0.5 * (pr + 1 - pr), np.where(d["y"] == 1, pr, 1 - pr))
        llr = np.bincount(d["voter"], weights=np.log(lh + 1e-9) - np.log(0.5), minlength=d["n_voters"])
        post = 1 / (1 + (1 - prior) / prior * np.exp(-np.clip(llr, -50, 50)))
        h = np.where(np.bincount(d["voter"], minlength=d["n_voters"]) == 0, prior, post)
    return th


METHODS = dict(naive=naive, elo=elo, capped=capped, robust=robust)
