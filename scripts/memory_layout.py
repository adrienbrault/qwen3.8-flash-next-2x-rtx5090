#!/usr/bin/env python3
"""Memory layout of the served GLM daily: VRAM per GPU and host DRAM, by category (2026-10-08, R900).

Everything is either exact (checkpoint tensor bytes from the safetensors headers, the served config) or measured
(nvidia-smi per GPU, the container's memory cgroup); what remains is reported as "other", never guessed.
  - weights by category from the checkpoint; routed experts (incl. the MTP layer's when MTP is on) split GPU/host by the
    served CPU split N of 288; the vision tower is host-side under vision_offload;
  - KV pool from the cache layout: per token per full-indexer DSA layer 1,120 B at 8,8 (latent 512 + scales 32 +
    fp16 indexer plane 512 + pooled keys 64), 11 layers; the index ring (INDEX_RING=1) drops the 512 B plane;
  - VRAM other = measured used - listed GPU items (CUDA contexts, graph pools, scratch, recurrent state, allocator);
  - host: container memory cgroup (anon + file) vs listed host items (CPU experts, embedding, recurrent cache).
usage: memory_layout.py --model <ckpt dir> --config <served config.yml> --container glm53 --out memory.json [--ring 0|1]
Runs on flan's host python (stdlib only).
"""
import argparse, json, re, struct, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model'); ap.add_argument('--config')
ap.add_argument('--container', default='glm53'); ap.add_argument('--out')
ap.add_argument('--ring', type=int, default=0); ap.add_argument('--dsa-layers', type=int, default=11)
# Qwen3.8-Flash-Next (R917): the KV page pool's bytes per token as measured for its cache (R579: 15,236 B at 8,8, all
# full-attention layers and the QSA indexer); overrides the GLM DSA formula when given
ap.add_argument('--kv-bytes-per-token', type=int, default=0)
# --rederive REC: recompute the host categories of a recorded memory.json with this version's rules (no new measurement:
# uses the record's cgroup bytes and checkpoint bytes), write it back and exit
ap.add_argument('--rederive', default='')
a = ap.parse_args()
if a.rederive:
    rec = json.load(open(a.rederive)); cg = rec['host_cgroup_bytes']; ck = rec['checkpoint_bytes_by_category']
    old = rec['host_categories_bytes']; rc_mb = rec['inputs'].get('sysmem_recurrent_cache_cap_mb')
    if rc_mb is None:
        rc_mb = int(next(re.search(r'configured (\d+) MB', k).group(1) for k in old if 'recurrent-state cache' in k))
    host = {k: v for k, v in old.items() if k.startswith(('CPU experts', 'embedding', 'vision tower'))}
    host[f'other: runtime, staging, buffers, recurrent-state cache (cap {rc_mb} MB)'] = \
        max(0, cg.get('anon', 0) + cg.get('shmem', 0) - sum(host.values()))
    rec['host_categories_bytes'] = host; rec['inputs']['sysmem_recurrent_cache_cap_mb'] = rc_mb
    Path(a.rederive).write_text(json.dumps(rec, indent=2) + '\n')
    for k, v in host.items(): print(f'  {k:70s} {v / 2 ** 30:7.2f} GiB')
    raise SystemExit(0)
GiB = 2 ** 30
model = Path(a.model)
IS_GLM = 'glm' in model.name.lower()  # GLM-5.3-Flash keeps its MTP layer as layers.45; Flash-Next as mtp.*
idx = json.load(open(model / 'model.safetensors.index.json'))['weight_map']
by_file = {}
for k, f in idx.items():
    by_file.setdefault(f, []).append(k)
cat = {}
def add(c, b): cat[c] = cat.get(c, 0) + b
for f, keys in by_file.items():
    with open(model / f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]; h = json.loads(fh.read(n))
    for k in keys:
        s, e = h[k]['data_offsets']; b = e - s
        m = re.search(r'layers\.(\d+)\.', k)
        layer = int(m.group(1)) if m else None
        if 'visual' in k: add('vision', b)
        elif 'embed_tokens' in k: add('embedding', b)
        elif 'lm_head' in k: add('lm_head', b)
        elif (IS_GLM and layer is not None and layer >= 45) or k.startswith('mtp.'):  # GLM: layer 45; Flash-Next: mtp.*
            add('mtp_experts' if '.mlp.experts.' in k else 'mtp_head', b)
        elif '.mlp.experts.' in k: add('routed_experts', b)
        elif 'shared_expert' in k: add('shared_experts', b)
        else: add('attention_norms_other', b)
cfg = Path(a.config).read_text()
def yval(key, default=None):
    m = re.search(rf'^\s*{key}:\s*(\S+)', cfg, re.M)
    return m.group(1) if m else default
cache_tokens = int(yval('cache_size', 0) or 0)
split_n = int(yval('cpu_moe_split_experts', 0) or 0)
rc_mb = int(yval('sysmem_recurrent_cache', 0) or 0)
draft = (yval('draft_mode', 'disabled') or 'disabled') != 'disabled'
vision_on = re.search(r'^\s*vision:\s*true', cfg, re.M) is not None
# vision_offload (VISION_OFFLOAD=1): the vision tower's weights stay in pinned host RAM
vision_host = vision_on and re.search(r'^\s*vision_offload:\s*true', cfg, re.M) is not None
if a.kv_bytes_per_token:
    per_tok, kv_layers = a.kv_bytes_per_token, 1
else:
    per_tok, kv_layers = 1120 - (512 if a.ring else 0), a.dsa_layers
kv = cache_tokens * per_tok * kv_layers
n_routed = 288
m = re.search(r'"(?:num_experts|n_routed_experts)":\s*(\d+)', (model / 'config.json').read_text())
if m: n_routed = int(m.group(1))
# the MTP layer's routed experts are split by the same N (MTP_FAST registers layer 45 with the trunk CPU worker)
mtp_exp = cat.get('mtp_experts', 0) if draft else 0
exp_host = (cat['routed_experts'] + mtp_exp) * split_n / n_routed
exp_gpu = cat['routed_experts'] + mtp_exp - exp_host
gpu = {'routed experts on GPU': exp_gpu, 'attention, norms, other weights': cat['attention_norms_other'],
       'shared experts': cat['shared_experts'], 'lm_head': cat['lm_head'],
       f'KV pool ({cache_tokens:,} tokens, 8-bit)': kv}
if vision_on and not vision_host: gpu['vision tower'] = cat['vision']
if draft: gpu['MTP layer (attention, shared expert, norms)'] = cat.get('mtp_head', 0)
smi = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                     capture_output=True, text=True, check=True).stdout.strip().splitlines()
used = {int(r.split(',')[0]): int(r.split(',')[1]) * 2 ** 20 for r in smi}
total = {int(r.split(',')[0]): int(r.split(',')[2]) * 2 ** 20 for r in smi}
gpu_used = sum(used.values())
gpu['other: CUDA contexts, graph pools, scratch, recurrent state, allocator'] = gpu_used - sum(gpu.values())
cid = subprocess.run(['sudo', '-n', 'docker', 'inspect', a.container, '--format', '{{.Id}}'], capture_output=True, text=True).stdout.strip()
mem = {}
for p in (f'/sys/fs/cgroup/system.slice/docker-{cid}.scope/memory.stat', f'/sys/fs/cgroup/docker/{cid}/memory.stat'):
    try:
        mem = dict((l.split()[0], int(l.split()[1])) for l in open(p)); break
    except OSError:
        pass
# anon (heap, pinned arena if anonymous) + shmem (memfd arena); file = page cache of the checkpoint reads, reported apart
host_used = mem.get('anon', 0) + mem.get('shmem', 0)
host = {f'CPU experts ({split_n} of {n_routed} per layer)': exp_host} if split_n else {}
# The recurrent-state cache is a cap that fills on demand (R917: Flash-Next's 4,096 MB cap exceeded the container's
# whole anon+shmem after warmup), so it is not listed as resident; whatever it holds is inside "other".
host |= {'embedding table': cat['embedding']}
if vision_host: host['vision tower (vision_offload)'] = cat['vision']
host[f'other: runtime, staging, buffers, recurrent-state cache (cap {rc_mb} MB)'] = max(0, host_used - sum(host.values()))
memtotal = int(re.search(r'^MemTotal:\s+(\d+) kB', open('/proc/meminfo').read(), re.M).group(1)) * 1024
rec = {'gpu_used_bytes': used, 'gpu_total_bytes': total, 'gpu_categories_bytes': gpu, 'host_total_bytes': memtotal,
       'host_cgroup_bytes': {k: mem.get(k) for k in ('anon', 'file', 'shmem', 'kernel') if k in mem},
       'host_categories_bytes': host,
       'inputs': {'cache_tokens': cache_tokens, 'cpu_split_experts': split_n, 'index_ring': a.ring, 'draft': draft,
                  'vision': vision_on, 'vision_offload': vision_host, 'sysmem_recurrent_cache_cap_mb': rc_mb, 'kv_bytes_per_token_per_layer': per_tok, 'kv_layers': kv_layers, 'routed_experts_per_layer': n_routed},
       'checkpoint_bytes_by_category': cat}
Path(a.out).write_text(json.dumps(rec, indent=2) + '\n')
for name, d in (('GPU (both)', gpu), ('host', host)):
    print(name); [print(f'  {k:70s} {v / GiB:7.2f} GiB') for k, v in d.items()]
print('GPU used per device:', {k: round(v / GiB, 2) for k, v in used.items()})
