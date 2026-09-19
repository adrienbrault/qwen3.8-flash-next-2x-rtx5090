# R556: adaptive MTP draft depth (round 2): −2.4 to −3.5 % on prose at 1 stream and gains nothing on code; not served

Results directory on the serving host: `results/2026-09-19-r556-adaptive-draft-r2`. Raw records: [`2026-09-19-r556-adaptive-draft-r2/`](2026-09-19-r556-adaptive-draft-r2/). Driver: [`scripts/r556-adaptive-draft-r2.sh`](../../scripts/r556-adaptive-draft-r2.sh). Date: 2026-09-19.

Configuration: the image of [R548](r548-promote-gdnbf16-ring.md) with a Python-only overlay (`EXL3_ADAPTIVE_DRAFT2=1`) that chooses the draft depth of each round from per-job, per-position acceptance estimates and an online table of round times. One boot per arm at a 1,015,808-token pool, NVMe tier off; `mp_decode.py`, 24 code and 24 prose prompts, 512 tokens, paired against arm S (served policy `[[4, 3], [8, 1]]`: code 197.3, prose 195.8 t/s at 1 stream; 125.8 / 118.0 per stream at 4).

| arm | code, 1 stream | prose, 1 stream | code, 4 streams | prose, 4 streams | mean depth per request |
| --- | --- | --- | --- | --- | --- |
| A3: adaptive, ceiling 3 | −2.09 % (−5.01 to +1.01) | −3.48 % (−5.72 to −1.27) | −0.45 % | −1.35 % (−2.27 to −0.33) | 2.15 (1.42–3.00) |
| A4: adaptive, ceiling 4 at one job | −0.61 % (−4.04 to +2.80) | −2.37 % (−4.59 to −0.22) | −1.21 % | +0.72 % | 2.40 (1.03–4.00) |

The rule fixed before the run required prose at 1 stream above zero for A3 and code at 1 stream ≥ +3.0 % for A4; neither arm met it.

An adaptive controller whose choices include the fixed depth cannot lose to it if it chooses well, so the loss is in the controller. Its per-request logs show one A4 request drafting 1 token in 73 of its 240 rounds while its own conditional acceptance per position read 0.66 / 0.995 / 0.78 / 0.97, which puts depth 4 at about 3.3 expected tokens per round against 1.7 at depth 1. At 1 stream a verify of 2 to 5 rows costs about the same time, so a shallow round saves little and loses tokens; the online round-time table, timed on the host, is the likely source of the shallow choices. The driver's in-unit comparison printed nothing because `mp_decode.py compare` strips trailing digits from tags and the arms were tagged `A31` / `A41`; the table above applies the same paired statistic per exact tag (`analysis.txt`).
