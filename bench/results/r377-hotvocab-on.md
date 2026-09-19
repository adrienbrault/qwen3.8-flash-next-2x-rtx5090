# R377: #303 MTP hot vocabulary is inapplicable on this box, by construction

Results directory on the serving host: `results/2026-09-16-r377-hotvocab-on`. Driver: [`scripts/r377-hotvocab-on.sh`](../../scripts/r377-hotvocab-on.sh). Date: 2026-09-16.


The port was rebased, built, and booted, and the engine refused it with its own guard:

```
File "exllamav3/architecture/qwen4_exp_mtp.py", line 178, in attach_to
    raise ValueError("MTP hot vocabulary requires target and draft on the same single GPU")
```

The served configuration is a **two-card layer split** (`gpu_split: [30, 30]`, `tensor_parallel: false`), and the feature
requires the target model and the MTP draft head on one GPU. So this is not a build problem, a patch problem, or an
experiment that needs better parameters: **the lever does not exist for this serving configuration.** The only
configuration in which it could be measured is single-GPU serving, which leaves the second card idle.

Two consequences of this arm. The first is that the failure surfaced as `docker run FAILED` and nothing
else for three separate attempts, because the launcher's run command ended in `>/dev/null 2>&1`; the error was captured
and printed only after that was fixed, and it named the cause immediately. The second is that the arm's container then
crash-looped under `--restart unless-stopped` while the launcher waited for readiness, and the runner's TERM trap
restored nothing — leaving the box on an experiment image until a restore was run by hand. That is the same lifecycle
gap the review's F3 describes, and it is the reason the last two experiments this session both ended with an explicit
restore rather than with the arm's own cleanup.
