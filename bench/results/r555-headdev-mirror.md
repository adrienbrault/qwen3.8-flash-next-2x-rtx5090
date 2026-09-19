# R555: the MTP draft's embedding copy on cuda:0 is bit-exact, +16,384 pool tokens, −1.8 % / −2.0 % decode at 1 stream; not served

Results directory on the serving host: `results/2026-09-19-r555-headdev-mirror`. Raw records: [`2026-09-19-r555-headdev-mirror/`](2026-09-19-r555-headdev-mirror/). Driver: [`scripts/r555-headdev-mirror.sh`](../../scripts/r555-headdev-mirror.sh). Date: 2026-09-19.

Configuration: the served configuration of [R548](r548-promote-gdnbf16-ring.md) with the 320 MiB copy of the 65,536 embedding rows the MTP draft head can emit moved from cuda:1 (which limits the pool) to cuda:0 (which keeps about 1.2 GiB unused). A first build failed on a container-store write error at 95 % disk use.

- Reference at 1,032,192 tokens: free VRAM at boot 2,013 / 737 MiB, minimum under a cold 120k prefill plus 4 streams 1,243 / 197 MiB.
- With the copy on cuda:0: 1,048,576 tokens boot with the same greedy output, free VRAM at boot 1,593 / 897 MiB, minimum under load 843 / 377 MiB; 1,064,960 does not boot ("Insufficient VRAM in split"). +16,384 tokens (+1.6 %), not the expected +32,768.
- `mp_decode.py`, 4 boots, 24 code and 24 prose prompts, 512 tokens, paired geometric mean: code 1 stream −1.83 % (95 % interval −2.21 to −1.48), prose 1 stream −2.04 % (−2.28 to −1.79; every prompt negative), code 4 streams +0.01 %, prose 4 streams −1.41 % (−2.86 to +0.23).

The draft token ids cross between the cards at every draft step. −2 % decode for +1.6 % pool is not served.
