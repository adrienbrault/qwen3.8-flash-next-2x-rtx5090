# R359: SWE-bench Verified, four subsets

Results directory on the serving host: `results/2026-09-16-r359-swebench`. Driver: [`scripts/r359-swebench.sh`](../../scripts/r359-swebench.sh). Date: 2026-09-16.


Agentic coding, which neither GSM8K nor tool-eval measures. Harness: mini-SWE-agent 2.4.6, the builtin
`benchmarks/swebench.yaml` (the leaderboard's bash-only setting, step_limit 250), `--subset verified --split test`,
scored by the official swebench harness in the official task images. The sampler reaches the server through this
seat's preset, because mini-swe sends only `max_tokens`.

**How the subsets were chosen, and what that forbids.** Each subset is drawn from a prior 500-instance scored run on
this box and stratified by that run's per-instance outcomes; none of them is a random sample of SWE-bench Verified.
The rates below therefore describe these instances only, and must not be read against a full-run rate. **A first
attempt read the "first 10" from `preds.json`, which is ordered by completion rather than by dataset order, and would
have executed the wrong instances.** The list each run executed is recorded in its results directory.

**Result — `2026-09-16-r359-swebench-10`, the dataset's first ten instances (all astropy):** resolved **10 of 10**.
All ten trajectories ended `Submitted` with a non-empty patch, at 45–163 steps; the 250-step limit was never reached,
so nothing was truncated by the harness. n=10 and one repository, so the subset cannot resolve a few points.

**Result — `2026-09-16-r360-swebench-strat`, 18 instances across six repositories, officially scored:** resolved
**17/18**. All 18 trajectories ended `Submitted` with a non-empty patch, 34–143 steps.

| repo | resolved |
| --- | --- |
| django | 2/3 |
| matplotlib | 3/3 |
| pydata | 3/3 |
| scikit-learn | 3/3 |
| sphinx-doc | 3/3 |
| sympy | 3/3 |
| **total** | **17/18** |

**Result — `2026-09-16-r361-swebench-failed`, ten instances selected because the prior scored run failed them after
submitting a patch** (drawn from its 113 unresolved, filtered to the 109 that ended `Submitted`, so the selection is
on capability failures rather than budget ones)**:** resolved **9/10**. All ten trajectories ended `Submitted`,
0 errors, 0 empty patches.

GSM8K, tool-eval and SWE-bench measure three different things on this stack; only SWE-bench measures agentic coding.

### All four subsets, de-duplicated

The subsets are **not disjoint**, and adding their totals gives the wrong count. Four runs cover 68 instance-runs but
only **49 unique instances**: the 30-instance subset re-ran all 18 of the stratified subset and one of the ten
selected from prior failures.

| subset | n | resolved | configuration |
| --- | --- | --- | --- |
| dataset's first ten (astropy) | 10 | **10** | pre-enablement |
| stratified, six repositories | 18 | **17** | pre-enablement |
| ten selected from prior failures | 10 | **9** | pre-enablement |
| 30-instance stratified, six repos | 30 | **28** | enabled (`tabbyapi:qsa-cid` + policy) |
| **unique instances** | **49** | **46** | mixed, see below |

**46 of 49 unique instances resolved.** Every subset was selected on a prior run's outcomes, so this is a descriptive
rate on these 49 instances and not an estimate of a rate on SWE-bench Verified. An earlier version of this section
attached a p-value to it: a sign test assumes selection independent of outcome, which does not hold here. An
inferential claim needs a held-out subset selected independently of any engine's results, which has not been run.

**The 19 repeated instances are the control for the enablement.** The stratified 18 were run before the enablement
and again inside the 30, and the one overlap with the failure-selected ten likewise: resolved 17 → 17 and 1 → 1,
**zero outcomes changed**. The configuration now served was therefore measured, rather than assumed, to be
quality-neutral on those 19 instances, independent of the byte-identity gates. That is weaker than a general
equivalence claim and is all it shows.
