import sys, pathlib, unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from arenarank.sim import simulate
from arenarank.fit import naive, robust, capped_weights
from arenarank.run import SCEN, score


class T(unittest.TestCase):
    def test_simulation_is_reproducible(self):
        a, b = simulate(3), simulate(3)
        self.assertTrue((a["y"] == b["y"]).all())

    def test_cap_bounds_each_voters_total_weight(self):
        d = simulate(1)
        w = capped_weights(d, np.ones(len(d["y"])), 10)
        self.assertLessEqual(np.bincount(d["voter"], weights=w).max(), 10 + 1e-9)

    def test_clean_votes_recover_the_ranking(self):
        d = simulate(2, **SCEN["clean"])
        self.assertGreater(score(d, naive(d))["rho"], 0.9)
        self.assertGreater(score(d, robust(d))["rho"], 0.9)

    def test_robust_resists_heavy_shills_better_than_naive(self):
        shifts = {"naive": 0, "robust": 0}
        for seed in range(8):
            d = simulate(seed, **SCEN["dev"])
            shifts["naive"] += score(d, naive(d))["shift"]
            shifts["robust"] += score(d, robust(d))["shift"]
        self.assertLess(shifts["robust"], shifts["naive"])


if __name__ == "__main__":
    unittest.main()
