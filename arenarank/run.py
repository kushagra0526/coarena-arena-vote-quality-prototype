"""python -m arenarank.run <first_seed> <n_seeds> <scenario> [out.json]   scenario: dev | held"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
from .sim import simulate
from .fit import METHODS

SCEN = dict(
    dev=dict(mix=dict(honest=0.7, random=0.1, lefty=0.1, shill=0.1), pos_bias=0.0, shill_act=8.0),
    # held-out: more shills, honest voters who also lean left, tie-prone voters, heavier tail, fewer votes
    held=dict(mix=dict(honest=0.45, random=0.1, lefty=0.1, shill=0.2, tier=0.15), pos_bias=0.3, tail=0.9, n_votes=4000, shill_act=0.4),
    clean=dict(mix=dict(honest=1.0), pos_bias=0.0),
)


def ranks(th):
    return np.argsort(np.argsort(-th))


def score(d, th):
    t = d["target"]
    return dict(rho=spearmanr(th, d["theta"])[0], top3=len(set(np.argsort(-th)[:3]) & {0, 1, 2}) / 3,
                shift=int(ranks(d["theta"])[t] - ranks(th)[t]))          # >0: the target is placed higher than it deserves


def run(first, n, scen):
    out = {m: [] for m in METHODS}
    for seed in range(first, first + n):
        d = simulate(seed, **SCEN[scen])
        for m, f in METHODS.items():
            out[m].append(score(d, f(d)))
    return out


def summary(out):
    return {m: dict(rho=float(np.mean([x["rho"] for x in v])), top3=float(np.mean([x["top3"] for x in v])),
                    target_shift=float(np.mean([x["shift"] for x in v])), shifted_ge2=int(sum(x["shift"] >= 2 for x in v)), n=len(v))
            for m, v in out.items()}


if __name__ == "__main__":
    first, n, scen = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    S = summary(run(first, n, scen))
    print(f"scenario {scen}, seeds {first}..{first + n - 1}")
    print("  method   rank corr   top-3 hit   shilled agent moved up (mean places)   seeds moved up >= 2")
    for m, s in S.items():
        print(f"  {m:8s} {s['rho']:9.3f} {s['top3']:11.2f} {s['target_shift']:20.2f} {s['shifted_ge2']:22d}/{s['n']}")
    if len(sys.argv) > 4:
        json.dump(S, open(sys.argv[4], "w"), indent=1)
