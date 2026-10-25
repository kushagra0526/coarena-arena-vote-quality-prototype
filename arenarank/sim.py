"""Simulated blind arena: M agents with known skill, V voters of mixed quality, battles judged left-vs-right.

Voter kinds: honest, random (coin flip), lefty (picks the left side 85%), shill (always votes for one target agent when it is in the battle,
otherwise behaves like random), tier (honest votes, but ties 40% of the time).
A battle outcome is y = 1 (left wins), 0 (right wins) or 0.5 (tie).
"""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(seed, n_models=12, n_voters=400, n_votes=6000, mix=None, beta=1.0, pos_bias=0.0, tail=1.2, target_rank=7, shill_act=1.0):
    """mix: dict of kind -> share of voters. Returns dict with votes arrays and ground truth."""
    r = np.random.default_rng(seed)
    mix = mix or dict(honest=0.7, random=0.1, lefty=0.1, shill=0.1)
    theta = np.sort(r.normal(0, 1, n_models))[::-1]            # theta[0] is the best agent
    kinds = r.choice(list(mix), size=n_voters, p=np.array(list(mix.values())) / sum(mix.values()))
    activity = r.pareto(tail, n_voters) + 1                    # heavy tail: a few voters cast very many votes
    activity = np.where(kinds == "shill", activity * shill_act, activity)
    voter = r.choice(n_voters, size=n_votes, p=activity / activity.sum())
    left = np.empty(n_votes, int); right = np.empty(n_votes, int)
    for k in range(n_votes):
        left[k], right[k] = r.choice(n_models, size=2, replace=False)
    target = target_rank                                       # shills push a mid-table agent
    y = np.empty(n_votes)
    u = r.random(n_votes)
    for k in range(n_votes):
        kind = kinds[voter[k]]
        p_left = sigmoid(beta * (theta[left[k]] - theta[right[k]]) + pos_bias)
        if kind == "honest":
            y[k] = float(u[k] < p_left)
        elif kind == "tier":
            y[k] = 0.5 if u[k] < 0.4 else float(r.random() < p_left)
        elif kind == "random":
            y[k] = float(u[k] < 0.5)
        elif kind == "lefty":
            y[k] = float(u[k] < 0.85)
        else:                                                  # shill
            if left[k] == target:
                y[k] = 1.0
            elif right[k] == target:
                y[k] = 0.0
            else:
                y[k] = float(u[k] < 0.5)
    return dict(theta=theta, kinds=kinds, voter=voter, left=left, right=right, y=y, target=target, n_models=n_models, n_voters=n_voters)
