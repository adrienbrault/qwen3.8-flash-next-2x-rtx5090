# decodekinds-r1 summary (20261116T102550Z-decodekinds-r1)

Client rates here are `(n - 1) / (latency - TTFT)`. They overstate answers of a few decode steps: the stream's last frame arrives ~0.1 ms after the one before it, so one step's tokens are counted without its time (+43 % on 26-token translations, +10-15 % on MBPP and extraction, 0-3 % above 150 tokens). The published per-category rates are step-based: generated tokens / decode steps from `server-steps.jsonl` (server_steps.py) divided by the mean inter-frame interval (bench/plot.py).

## pass A

- reasoning (thinking on, 1,024 forced, c1) code: median 251.9 t/s (n=3)
- reasoning (thinking on, 1,024 forced, c1) prose: median 252.2 t/s (n=3)
- MiaAI coding (thinking off, c1, 5 x cap 400): median 400.2 t/s, tokens [59, 59, 59, 59, 59], finish ['stop'], <think> in text 0, accepted/rejected drafts 210/45
- MiaAI structured (thinking off, c1, 5 x cap 400): median 393.3 t/s, tokens [400, 400, 400, 400, 400], finish ['length'], <think> in text 0, accepted/rejected drafts 1500/0
- MiaAI prose (thinking off, c1, 5 x cap 400): median 305.5 t/s, tokens [400, 400, 400, 400, 400], finish ['length'], <think> in text 0, accepted/rejected drafts 1350/600

### vllm bench, thinking off, natural length (cap 1,024), c1: 560 ok, 0 failed, reasoning frames 0

| category | n | decode t/s per stream (token-weighted) | mean output tokens | TTFT p50 ms | finish | drafts accepted / (accepted + rejected) |
|---|---:|---:|---:|---:|---|---:|
| coding | 10 | 328.4 | 512 | 109 | {'stop': 9, 'length': 1} | 0.80 |
| extraction | 10 | 374.4 | 65 | 182 | {'stop': 10} | 0.82 |
| humanities | 10 | 248.4 | 975 | 67 | {'length': 7, 'stop': 3} | 0.53 |
| math | 10 | 349.2 | 439 | 113 | {'stop': 10} | 0.87 |
| math_reasoning | 80 | 344.6 | 306 | 134 | {'stop': 80} | 0.84 |
| mbpp | 80 | 344.3 | 94 | 138 | {'stop': 79, 'length': 1} | 0.76 |
| qa | 80 | 267.9 | 348 | 60 | {'stop': 77, 'length': 3} | 0.59 |
| rag | 80 | 306.9 | 150 | 250 | {'stop': 80} | 0.70 |
| reasoning | 10 | 299.2 | 284 | 118 | {'stop': 10} | 0.70 |
| roleplay | 10 | 260.3 | 541 | 135 | {'stop': 8, 'length': 2} | 0.57 |
| stem | 10 | 281.2 | 881 | 125 | {'length': 3, 'stop': 7} | 0.64 |
| summarization | 80 | 272.8 | 247 | 244 | {'stop': 80} | 0.60 |
| translation | 80 | 403.5 | 26 | 119 | {'stop': 80} | 0.65 |
| writing | 10 | 242.4 | 430 | 119 | {'stop': 8, 'length': 2} | 0.51 |
| **all** | 560 | 292.0 | | | | |

### vllm bench, thinking off, natural length (cap 1,024), c4: 560 ok, 0 failed, reasoning frames 0

| category | n | decode t/s per stream (token-weighted) | mean output tokens | TTFT p50 ms | finish | drafts accepted / (accepted + rejected) |
|---|---:|---:|---:|---:|---|---:|
| coding | 10 | 155.8 | 527 | 206 | {'stop': 9, 'length': 1} | 0.81 |
| extraction | 10 | 191.8 | 65 | 269 | {'stop': 10} | 0.82 |
| humanities | 10 | 118.7 | 972 | 210 | {'length': 7, 'stop': 3} | 0.55 |
| math | 10 | 169.5 | 476 | 198 | {'stop': 10} | 0.86 |
| math_reasoning | 80 | 167.7 | 305 | 229 | {'stop': 80} | 0.85 |
| mbpp | 80 | 161.1 | 92 | 228 | {'stop': 80} | 0.76 |
| qa | 80 | 127.0 | 361 | 148 | {'stop': 76, 'length': 4} | 0.60 |
| rag | 80 | 145.1 | 157 | 310 | {'stop': 80} | 0.69 |
| reasoning | 10 | 141.5 | 287 | 205 | {'stop': 10} | 0.69 |
| roleplay | 10 | 123.3 | 557 | 227 | {'stop': 8, 'length': 2} | 0.55 |
| stem | 10 | 136.2 | 886 | 212 | {'length': 4, 'stop': 6} | 0.63 |
| summarization | 80 | 130.0 | 250 | 310 | {'stop': 80} | 0.60 |
| translation | 80 | 159.9 | 26 | 209 | {'stop': 80} | 0.66 |
| writing | 10 | 116.7 | 459 | 206 | {'stop': 8, 'length': 2} | 0.51 |
| **all** | 560 | 139.1 | | | | |

- R828 code edit (thinking off, 2,048 forced, c1, 36 measured): per-stream median 384.5 t/s (min 379.2, max 388.6)

## pass B

- reasoning (thinking on, 1,024 forced, c1) code: median 249.9 t/s (n=3)
- reasoning (thinking on, 1,024 forced, c1) prose: median 252.3 t/s (n=3)
- MiaAI coding (thinking off, c1, 5 x cap 400): median 399.9 t/s, tokens [59, 59, 59, 59, 59], finish ['stop'], <think> in text 0, accepted/rejected drafts 210/45
- MiaAI structured (thinking off, c1, 5 x cap 400): median 393.2 t/s, tokens [400, 400, 400, 400, 400], finish ['length'], <think> in text 0, accepted/rejected drafts 1500/0
- MiaAI prose (thinking off, c1, 5 x cap 400): median 305.0 t/s, tokens [400, 400, 400, 400, 400], finish ['length'], <think> in text 0, accepted/rejected drafts 1350/600

### vllm bench, thinking off, natural length (cap 1,024), c1: 560 ok, 0 failed, reasoning frames 0

| category | n | decode t/s per stream (token-weighted) | mean output tokens | TTFT p50 ms | finish | drafts accepted / (accepted + rejected) |
|---|---:|---:|---:|---:|---|---:|
| coding | 10 | 328.3 | 512 | 110 | {'stop': 9, 'length': 1} | 0.80 |
| extraction | 10 | 375.8 | 65 | 190 | {'stop': 10} | 0.82 |
| humanities | 10 | 248.5 | 975 | 66 | {'length': 7, 'stop': 3} | 0.53 |
| math | 10 | 348.9 | 439 | 113 | {'stop': 10} | 0.87 |
| math_reasoning | 80 | 344.9 | 306 | 135 | {'stop': 80} | 0.84 |
| mbpp | 80 | 343.4 | 94 | 138 | {'stop': 79, 'length': 1} | 0.76 |
| qa | 80 | 267.9 | 348 | 60 | {'stop': 77, 'length': 3} | 0.59 |
| rag | 80 | 307.2 | 150 | 250 | {'stop': 80} | 0.70 |
| reasoning | 10 | 301.4 | 284 | 119 | {'stop': 10} | 0.70 |
| roleplay | 10 | 259.8 | 541 | 135 | {'stop': 8, 'length': 2} | 0.57 |
| stem | 10 | 281.3 | 881 | 119 | {'length': 3, 'stop': 7} | 0.64 |
| summarization | 80 | 273.1 | 247 | 246 | {'stop': 80} | 0.60 |
| translation | 80 | 398.6 | 26 | 118 | {'stop': 80} | 0.65 |
| writing | 10 | 242.7 | 430 | 119 | {'stop': 8, 'length': 2} | 0.51 |
| **all** | 560 | 292.1 | | | | |

### vllm bench, thinking off, natural length (cap 1,024), c4: 560 ok, 0 failed, reasoning frames 0

| category | n | decode t/s per stream (token-weighted) | mean output tokens | TTFT p50 ms | finish | drafts accepted / (accepted + rejected) |
|---|---:|---:|---:|---:|---|---:|
| coding | 10 | 149.4 | 518 | 210 | {'stop': 9, 'length': 1} | 0.79 |
| extraction | 10 | 161.0 | 65 | 281 | {'stop': 10} | 0.82 |
| humanities | 10 | 118.4 | 986 | 174 | {'length': 7, 'stop': 3} | 0.55 |
| math | 10 | 170.5 | 459 | 216 | {'stop': 10} | 0.87 |
| math_reasoning | 80 | 165.4 | 306 | 226 | {'stop': 80} | 0.84 |
| mbpp | 80 | 163.9 | 95 | 229 | {'stop': 79, 'length': 1} | 0.76 |
| qa | 80 | 126.3 | 356 | 149 | {'stop': 77, 'length': 3} | 0.59 |
| rag | 80 | 145.4 | 151 | 309 | {'stop': 80} | 0.70 |
| reasoning | 10 | 141.6 | 290 | 208 | {'stop': 9, 'length': 1} | 0.69 |
| roleplay | 10 | 123.6 | 550 | 224 | {'stop': 8, 'length': 2} | 0.56 |
| stem | 10 | 137.3 | 890 | 207 | {'length': 4, 'stop': 6} | 0.64 |
| summarization | 80 | 128.6 | 249 | 315 | {'stop': 80} | 0.60 |
| translation | 80 | 155.3 | 26 | 209 | {'stop': 80} | 0.66 |
| writing | 10 | 117.6 | 448 | 210 | {'stop': 8, 'length': 2} | 0.52 |
| **all** | 560 | 138.3 | | | | |

- R828 code edit (thinking off, 2,048 forced, c1, 36 measured): per-stream median 382.2 t/s (min 377.2, max 388.1)

GPU max temperature: GPU 0 74 C, GPU 1 50 C
