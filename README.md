# arenarank: what a few bad voters do to a computer-use arena leaderboard

CoArena ranks agents from blind human votes with a Bradley-Terry fit (their paper, https://arxiv.org/abs/2609.14239, covers ties, confidence intervals, rank bands and vote quality via agreement statistics). This prototype is a simulator with known truth that asks one narrow question: **when some voters are shills, coin-flippers or left-clickers, how far does the leaderboard move, and what helps?**

Simulation: 12 agents with known skill, 400 voters, heavy-tailed voting activity, left/right sides randomised. Voter kinds: honest, random, lefty (85% left), shill (always votes for one mid-table target when it is in the battle), tie-prone.

Methods: `naive` (penalized BT, every vote counts), `elo` (streaming, lr 0.03), `capped` (BT, each voter's total weight capped at 15), `robust` (BT + a left-side term, per-voter honesty weight from EM against a coin-flip model, plus the cap).
Metrics: Spearman rank correlation with the true order, top-3 hit rate, and how many places the shilled agent is moved up.

## Results
Clean votes (30 seeds): naive 0.988, robust 0.988 rank correlation. The defence costs nothing when everyone is honest.

Dev, 10% shills with heavy activity (30 seeds):

| | rank corr | top-3 | shilled agent moved up | seeds moved up >= 2 |
|---|---|---|---|---|
| naive | 0.803 | 0.63 | 6.33 | 30/30 |
| elo | 0.748 | 0.66 | 6.33 | 30/30 |
| capped | 0.890 | 0.79 | 3.93 | 27/30 |
| robust | 0.933 | 0.84 | 1.93 | 16/30 |

Held-out, 40 seeds, scenario fixed in git before the first run: 20% shills spread over many light accounts (sybil), left-leaning honest voters, tie-prone voters, heavier tail, 4,000 votes:

| | rank corr | top-3 | shilled agent moved up | seeds moved up >= 2 |
|---|---|---|---|---|
| naive | 0.935 | 0.88 | 1.68 | 16/40 |
| elo | 0.918 | 0.85 | 1.30 | 15/40 |
| capped | 0.906 | 0.82 | 1.88 | 22/40 |
| robust | 0.921 | 0.87 | 1.18 | 16/40 |

Read: per-voter weighting works against a few accounts casting many votes. Against many light sybil accounts it barely helps (and costs a little rank correlation versus naive), because no single account has enough votes to look bad. That case needs a different signal (account age, device, pairwise vote graph), which this prototype does not have.

## Limits
- All simulated; the voter mixes are my guesses, not CoArena data. This says nothing about how clean their real votes are.
- No confidence intervals or rank bands here (the paper has them); only point estimates.
- Skill is one number per agent with a fixed noise level; real tasks vary by task type.
- Honest-voter noise (beta = 1) sets how much evidence each vote carries; other values change the numbers.

## Run
```
python -m unittest tests.test_arenarank
python -m arenarank.run 0 30 dev
python -m arenarank.run 100 40 held
```
numpy + scipy.

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with CoArena or Coasty.
