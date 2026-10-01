#!/usr/bin/env python3
"""R825p CPU gates, R825c frontend measurements, and immutable T32 smoke (stdlib only)."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import types

sys.dont_write_bytecode = True
import r823_reuse as reuse

IMAGE = 'tabbyapi:r825c-hostprepare'
HEADER = '# R825p (2026-10-01): gated R825c frontend-safe resumable whole-prompt prefill on the R823c daily.\n# See r825p-promote-wholeprompt.sh; 46 selectors, trace ON, NVMe OFF.\n'
OLD_MD5 = '8b644c17e60049a8069fc901b6f091fa'

# The served launcher carries the operator-recorded build ID literally: the daily must not depend on a file
# in an experiment directory (and the launcher is mirrored publicly). IMAGE_ID.env still pins the units.
PINNED_IMAGE_ID = 'sha256:aa04a1cbe94b60bbd91e3d69d15f5e1b6eb679f14885e599404de218d8a897c9'
IMAGE_GUARD = f"""# R825c image pin (operator-recorded build ID of tabbyapi:r825c-hostprepare).
R825C_IMAGE_ID={PINNED_IMAGE_ID}
"""
IMAGE_CHECK = """[ "$(sudo docker image inspect -f '{{.Id}}' "$IMG")" = "$R825C_IMAGE_ID" ] || { log 'ABORT: R825c image differs from operator pin'; exit 3; }
"""


def image_id(path):
    text = Path(path).read_text()
    match = re.fullmatch(r'R825C_IMAGE_ID=(sha256:[0-9a-f]{64})\n?', text)
    assert match, 'fill R825c IMAGE_ID.env after building (full sha256 image ID)'
    return match[1]


def candidate(old):
    assert hashlib.md5(old.encode()).hexdigest() == OLD_MD5, 'old launcher identity'
    new = old.replace('#!/usr/bin/env bash\n', '#!/usr/bin/env bash\n' + HEADER + IMAGE_GUARD, 1)
    new = new.replace('DAILY_IMG=tabbyapi:r823c-cachetail-inforward\n', f'DAILY_IMG={IMAGE}\n', 1)
    new = new.replace('sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }\n',
                      'sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }\n' + IMAGE_CHECK, 1)
    return re.sub(r'^(EXTRA_ENV=\$\{EXTRA_ENV:-.*)(\})$',
                  lambda m: m[1] + ' EXL3_PREFILL_WHOLE_PROMPT=1 EXL3_PREFILL_RESUMABLE=1' + m[2], new, count=1, flags=re.M)


def selectors(text):
    line = re.search(r'^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$', text, re.M)[1]
    entries = line.split()
    values = dict(x.split('=', 1) for x in entries)
    assert len(entries) == len(values) == 46, 'duplicate/missing launcher selectors'
    return values


def smoke_rows(fixtures):
    for name, want in [('manifest.json', '4b2b0bec52605fe6403b89bb10ad19d074cab359c9ce70cbe1bdeaf9c6b62475'),
                       ('payloads.json.gz', '21688f887161e59d721825a8d70821ae1c9aaf98b72e7ef9f3e04fd8c5c65726')]:
        assert hashlib.sha256((Path(fixtures) / name).read_bytes()).hexdigest() == want, 'immutable fixture drift: ' + name
    manifest, payloads = reuse.load_fixtures(fixtures)
    keys = ['r823/raw/T32/s0/0/seed', 'r823/raw/T32/s0/1/edit', 'r823/raw/T32/s0/1/hit']
    rows = [next(r for r in manifest['requests'] if r['key'] == k) for k in keys]
    assert rows[1]['payload'] == rows[2]['payload'], 'HIT must be the identical edited payload'
    assert rows[0]['n'] == rows[1]['n'] == rows[2]['n'] == 67618
    assert rows[1]['lcp'] == 60000
    return rows, payloads


def expected_checkpoint(policy_path, render, seed_n, lcp, base):
    # Execute the exact policy source exported from the served, landing-verified image.
    spec = importlib.util.spec_from_file_location('served_checkpoint_policy', policy_path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    gen = types.SimpleNamespace(max_chunk_size=render['chunk'],
                                recurrent_checkpoint_interval=render['policy_near'],
                                recurrent_checkpoint_interval_pp=render['policy_pp'],
                                recurrent_checkpoint_tail_pp=render['policy_tail'])
    end = seed_n - 1
    eligible = [pos for pos in range(base + 256, lcp // 256 * 256 + 1, 256)
                if pos >= end - gen.recurrent_checkpoint_tail_pp
                and (pos - base) % policy.interval(gen, pos, end) == 0]
    assert eligible, 'no eligible tail checkpoint'
    return max(eligible)


def check_smoke(fixtures, clients, text, policy_path):
    rows, _ = smoke_rows(fixtures)
    traces = reuse.parse_traces(text)
    joined = reuse.joins(traces, clients, rows)
    server = reuse.parse_server(text)
    for row in joined:
        render = row['trace']['render']
        assert (render['policy_pp'], render['policy_tail'], render['policy_near'],
                render['chunk'], render['policy_inforward']) == (4096, 12288, 2048, 2048, True)
        assert row['trace']['allocation']['disk'] == 'off', 'NVMe differs from measured D'
        sr = server[render['http_serial']]
        assert sr['n'] == row['fixture']['n']
        assert sr['n'] - sr['new'] == row['trace']['allocation']['selected'], 'server/trace cache mismatch'
        assert row['trace']['finish']['cached_tokens'] == sr['n'] - sr['new']
    seed, edit, hit = joined
    base = seed['trace']['allocation']['selected']
    assert base == 0, 'seed must be cold; do not seed a truncated prefix'
    expected = expected_checkpoint(policy_path, seed['trace']['render'], rows[0]['n'], rows[1]['lcp'], base)
    got = edit['trace']['allocation']['selected']
    assert got == expected, f'edit cached {got}, expected policy checkpoint {expected}'
    assert any(e['event'] == 'save' and e.get('position') == expected
               for e in seed['events']), 'expected checkpoint was not saved by seed'
    assert any(e['event'] == 'tail_capture' and e.get('stored')
               for e in seed['events']), 'no in-forward capture publication'
    assert not any(e['event'] == 'tail_fallback' for e in traces), 'unexpected tail fallback'
    cached_hit = hit['trace']['allocation']['selected']
    assert cached_hit >= rows[2]['n'] - 256, f'HIT cached {cached_hit} < N-256'
    result = dict(edit_cached=got, expected_from_served_policy=expected,
                  edit_page=rows[1]['lcp'] // 256 * 256, hit_cached=cached_hit,
                  hit_min=rows[2]['n'] - 256,
                  policy_sha256=hashlib.sha256(Path(policy_path).read_bytes()).hexdigest())
    return result


def smoke_client(fixtures, url, out):
    rows, payloads = smoke_rows(fixtures)
    preflight = []
    for h in dict.fromkeys(r['payload'] for r in rows):
        p = payloads[h]
        observed = reuse.post(url + '/token/encode', {'text': p['prompt'], 'add_bos_token': False})['tokens']
        assert observed == p['token_ids'], 'served tokenizer differs from immutable fixture'
        preflight.append(dict(payload=h, n=len(observed)))
    reuse.dump(str(out) + '.preflight.json', preflight)
    with open(out, 'w') as f:
        for row in rows:  # seed, first edit, immediate identical HIT; no intervening request
            result = reuse.request_one(row, payloads[row['payload']], url)
            f.write(json.dumps(result) + '\n')
            f.flush()
            assert result['status'] == 'VALID', result.get('error', 'invalid client result')


def check_env(template, inspect, config):
    obj = inspect[0]
    env = dict(x.split('=', 1) for x in obj['Config']['Env'])
    for k, v in selectors(template).items():
        assert env.get(k) == v, 'selector mismatch: ' + k
    assert env.get('EXL3_CACHE_TRACE') == '1', 'trace disabled'
    assert 'EXL3_NVME_TIER' not in env, 'NVMe tier unexpectedly on'
    assert obj['State']['Status'] == 'running'
    for mount in obj['Mounts']:
        assert not mount['Destination'].startswith(('/opt/venv/lib/python3.12/site-packages/exllamav3',
                                                     '/app/backends', '/app/endpoints')), 'source overlay mount'
    for line in ('cache_size: 901120', 'max_batch_size: 8', 'chunk_size: 2048',
                 'sysmem_recurrent_cache: 4096', 'sysmem_kv_cache: 0'):
        assert line in config, 'config mismatch: ' + line


NEEDLE_TOKENS = {131072: 105680, 240000: 193464}


def check_needles(rows, tag):
    want = {(n, f) for n in (131072, 240000) for f in (0.08, 0.3, 0.55, 0.8, 0.96)}
    assert len(rows) == len(want) and {(r['ctx_requested'], r['frac']) for r in rows} == want
    for r in rows:
        assert r['tag'] == tag and r['hit'] is True and not r.get('err'), 'needle miss/error'
        assert r.get('finish_reason') and r.get('completion_tokens', 0) > 0, 'incomplete needle'
        # fn_needle_oai sizes by words: the served daily's R810 run measured exactly these token counts
        # (ctx~131072 -> 105,680; ctx~240000 -> 193,464). The 0.9 x ctx bound failed R823p try 1 at 10/10 HIT.
        assert r['prompt_tokens'] == NEEDLE_TOKENS[r['ctx_requested']], 'needle depth mismatch'


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# Preserve the complete observed image/runtime environment, not only the launcher's 46 keys.
EVIDENCE = Path(__file__).with_name('r825p-fixtures')
_ENV_CHECK = check_env

def reference_env():
    lines = (EVIDENCE / 'live-container-env.txt').read_text().splitlines()
    entries = [x for x in lines if x]
    env = dict(x.split('=', 1) for x in entries)
    assert len(env) == len(entries), 'duplicate baseline environment key'
    return env


def check_env(template, inspect, config):
    _ENV_CHECK(template, inspect, config)
    entries = inspect[0]['Config']['Env']
    actual = dict(x.split('=', 1) for x in entries)
    expected = dict(reference_env(), EXL3_PREFILL_WHOLE_PROMPT='1', EXL3_PREFILL_RESUMABLE='1')
    assert len(entries) == len(actual), 'duplicate served env key'
    assert actual == expected, ('full environment drift', sorted(set(actual.items()) ^ set(expected.items())))


def check_old_env(inspect):
    entries = inspect[0]['Config']['Env']
    actual = dict(x.split('=', 1) for x in entries)
    assert len(entries) == len(actual) and actual == reference_env(), 'live baseline env drift'
    assert inspect[0]['State']['Status'] == 'running'
    for mount in inspect[0]['Mounts']:
        assert not mount['Destination'].startswith(('/opt/venv/lib/python3.12/site-packages/exllamav3',
                                                     '/app/backends', '/app/endpoints')), 'baseline source overlay'


def verify_evidence():
    for name, meta in json.loads((EVIDENCE / 'provenance.json').read_text()).items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == meta['sha256'], name


def token_count(event):
    """Actual TabbyAPI completion SSE: multiple MTP tokens can share a frame.

    Replay uses the real R823p decoded SSE events. Do not equate text frames with tokens.
    """
    assert not event.get('error'), 'SSE server error'
    count = 0
    for c in event.get('choices', []):
        lp = c.get('logprobs') or {}
        tokens = lp.get('tokens') or lp.get('content') or []
        visible = c.get('text') or (c.get('delta') or {}).get('content')
        assert tokens or not visible, 'token-bearing SSE missing requested logprobs'
        count += len(tokens)
    assert isinstance(count, int) and count >= 0
    return count


def read_sse(response):
    # Same data-line/event-boundary protocol as r823_reuse.sse, preserving wire frames.
    import time
    data = []
    for raw in response:
        line = raw.decode('utf-8').rstrip('\r\n')
        if line.startswith('data:'):
            data.append(line[5:].lstrip())
        elif not line and data:
            value = '\n'.join(data); data = []
            if value == '[DONE]':
                return
            event = json.loads(value)
            token_count(event)  # fail closed on error/missing token instrumentation
            yield time.time(), event
    raise ValueError('truncated SSE: no [DONE]')


def prepare(url, model, out):
    """Size with the REAL tokenizer, once; old and new send identical prompt bytes.

    A UUID repeated across the first page makes it unlike every earlier fixture/warmup.
    Frontend cached=0 is still mandatory. No inferred word-to-token depth thresholds.
    """
    import uuid
    salt = uuid.uuid4().hex
    prompts = {}
    for name, target in [('cold50', 50000), ('long90', 90000), ('health90', 90000), ('arrival90', 90000), ('short200', 200)]:
        head = (f'R825p {salt} {name}: independent cold passage.\n' * (2 if target == 200 else 32))
        line = 'Record: amber bridge copper forest window. Explain the next numbered record.\n'
        head_n = len(reuse.post(url + '/token/encode', {'text': head, 'add_bos_token': False})['tokens'])
        repeat = target // 16
        for _ in range(12):
            prompt = head + line * repeat + '\nSummarize the records in numbered statements.\n'
            ids = reuse.post(url + '/token/encode', {'text': prompt, 'add_bos_token': False})['tokens']
            if abs(len(ids) - target) <= (20 if target == 200 else 512):
                break
            repeat = max(1, round(repeat * (target - head_n) / max(1, len(ids) - head_n)))
        else:
            raise ValueError('real tokenizer sizing did not converge')
        prompts[name] = dict(prompt=prompt, n=len(ids), sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                             first_page_sha256=hashlib.sha256(json.dumps(ids[:256]).encode()).hexdigest())
    decode = ('R825p ' + salt + ' decode-only reference.\n') * 32
    decode += 'Continue a long numbered list of Python functions with explanations, at least 8192 tokens.\n1.'
    prompts['decode'] = dict(prompt=decode, sha256=hashlib.sha256(decode.encode()).hexdigest())
    prompts['model'] = model
    pages = [v['first_page_sha256'] for v in prompts.values() if isinstance(v, dict) and 'first_page_sha256' in v]
    assert len(set(pages)) == len(pages), 'first pages must be unique across all cold cases'
    reuse.dump(out, prompts)


def stream(url, model, prompt, key, ntokens, ready=None, stop=None, dispatched=None):
    import time
    import urllib.request
    body = dict(model=model, prompt=prompt, temperature=0, max_tokens=ntokens,
                min_tokens=ntokens, stream=True, stream_options={'include_usage': True},
                logprobs=1, loop_detect_window=0, add_bos_token=False)
    request = urllib.request.Request(url + '/completions', json.dumps(body).encode(),
        {'Content-Type': 'application/json', 'x-r823-request': key, 'x-r823-conversation': key})
    began = time.perf_counter()
    result = dict(key=key, dispatch_epoch=time.time(), dispatch_mono=began, frames=[], status='INVALID', cancelled=False)
    if dispatched is not None:
        dispatched.set()
    try:
        with reuse.opener().open(request, timeout=900) as response:
            for stamp, event in read_sse(response):
                result['frames'].append(dict(t=stamp, elapsed_s=time.perf_counter()-began, event=event, tokens=token_count(event)))
                if ready is not None and sum(f['tokens'] > 0 for f in result['frames']) >= 8:
                    ready.set()
                if stop is not None and stop.is_set() and result['frames'][-1]['tokens']:
                    result['cancelled'] = True
                    break
        result['end_epoch'] = time.time()
        result['elapsed_s'] = time.perf_counter()-began
        if result['cancelled']:
            result['status'] = 'VALID'
        else:
            events = [f['event'] for f in result['frames']]
            out, _ = reuse.output(events)
            usage = [e['usage'] for e in events if e.get('usage')]
            assert usage and out['finish'], 'missing usage/finish'
            assert usage[-1]['completion_tokens'] == ntokens, 'forced generation length mismatch'
            assert sum(f['tokens'] for f in result['frames']) == ntokens, 'SSE token count != usage'
            result.update(status='VALID', usage=usage[-1])
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        if ready is not None:
            ready.set()  # unblock caller on failure; VALID is still required
    return result


def frontend_client(url, prompts_path, phase, out):
    import concurrent.futures
    import threading
    import time
    prompts = json.loads(Path(prompts_path).read_text())
    # Tokenizer/render changes cannot silently alter either target context size.
    for name in ('cold50', 'long90', 'health90', 'arrival90', 'short200'):
        n = len(reuse.post(url + '/token/encode', {'text': prompts[name]['prompt'],
                                                'add_bos_token': False})['tokens'])
        assert n == prompts[name]['n'], 'candidate tokenizer drift'
    result = dict(phase=phase, prompts=prompts)
    cold = stream(url, prompts['model'], prompts['cold50']['prompt'], f'r823/r825p/{phase}/cold50', 32)
    result['cold50'] = cold
    reuse.dump(out, result)
    assert cold['status'] == 'VALID', cold.get('error')
    ready, stop = threading.Event(), threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        decoder = pool.submit(stream, url, prompts['model'], prompts['decode']['prompt'],
                              f'r823/r825p/{phase}/decode', 8192, ready, stop)
        try:
            assert ready.wait(120), 'decode did not reach eight token frames before long request'
            if decoder.done():
                raise ValueError('decode ended before concurrent long request: ' + str(decoder.result()))
            result['long90'] = stream(url, prompts['model'], prompts['long90']['prompt'],
                                      f'r823/r825p/{phase}/long90', 32)
            # Obtain a post-prefill token to bound the entire gap, including a zero-token stall.
            time.sleep(0.5)
        finally:
            stop.set()
        result['decode'] = decoder.result(timeout=120)
    reuse.dump(out, result)
    time.sleep(1)  # permit server-side disconnect cancellation to publish final trace
    assert result['long90']['status'] == result['decode']['status'] == 'VALID', 'concurrency client failed'

    # Each scenario uses a distinct first page; all arms reuse this same artifact.
    # Independent client threads continue issuing health probes while the SERVER loop stalls.
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        dispatched = threading.Event()
        long_health = pool.submit(stream, url, prompts['model'], prompts['health90']['prompt'],
                                  f'r823/r825p/{phase}/health90', 32, dispatched=dispatched)
        assert dispatched.wait(10), 'health long request not dispatched'
        polls = []
        deadline = time.perf_counter()
        while not long_health.done():
            polls.append(pool.submit(health_once, url.removesuffix('/v1') + '/health'))
            deadline += 0.2
            time.sleep(max(0, deadline-time.perf_counter()))
        result['health90'] = long_health.result()
        result['health_samples'] = [f.result() for f in polls]
    reuse.dump(out, result)
    assert result['health90']['status'] == 'VALID', 'health cold request failed'
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        dispatched = threading.Event()
        arrival = pool.submit(stream, url, prompts['model'], prompts['arrival90']['prompt'],
                              f'r823/r825p/{phase}/arrival90', 32, dispatched=dispatched)
        assert dispatched.wait(10), 'late-arrival long request not dispatched'
        time.sleep(1.0)
        result['short200'] = stream(url, prompts['model'], prompts['short200']['prompt'],
                                     f'r823/r825p/{phase}/short200', 16)
        result['arrival90'] = arrival.result(timeout=900)
    reuse.dump(out, result)
    assert result['arrival90']['status'] == result['short200']['status'] == 'VALID', 'late arrival failed'


def health_once(url):
    import socket
    import time
    import urllib.error
    import urllib.request
    began = time.perf_counter()
    result = dict(dispatch_epoch=time.time(), status='ERROR', timeout=False)
    try:
        # New connections match drain-gate.py's urllib client (no proxy, no keepalive).
        with reuse.opener().open(urllib.request.Request(url, method='GET'), timeout=3.0) as response:
            response.read()
            result.update(status='OK' if response.status == 200 else 'ERROR', http_status=response.status)
    except (TimeoutError, socket.timeout) as exc:
        result.update(status='TIMEOUT', timeout=True, error=str(exc))
    except urllib.error.HTTPError as exc:
        result.update(status='ERROR', http_status=exc.code, error=str(exc))
        exc.close()
    except urllib.error.URLError as exc:
        timed_out = isinstance(exc.reason, (TimeoutError, socket.timeout))
        result.update(status='TIMEOUT' if timed_out else 'ERROR', timeout=timed_out, error=str(exc))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result.update(latency_s=time.perf_counter()-began, end_epoch=time.time())
    return result


def health_metrics(samples, start, end):
    import math
    assert samples and end > start
    assert all(math.isfinite(s[k]) for s in samples for k in ('dispatch_epoch','latency_s','end_epoch'))
    assert all(s['latency_s'] >= 0 and s['status'] in ('OK','TIMEOUT','ERROR') for s in samples)
    assert all(s['timeout'] == (s['status'] == 'TIMEOUT') for s in samples)
    stamps = [s['dispatch_epoch'] for s in samples]
    assert all(0 < b-a <= .4 for a,b in zip(stamps, stamps[1:])), 'health polling cadence invalid'
    during = [s for s in samples if start <= s['dispatch_epoch'] <= end]
    assert len(during) >= 5, 'too few health probes during prefill'
    assert during[0]['dispatch_epoch'] <= start+.4 and during[-1]['dispatch_epoch'] >= end-.4, 'health probes do not cover prefill'
    # Include every poll through completion, even the poll whose timeout ends after prefill.
    return dict(longest_latency_s=max(s['latency_s'] for s in samples),
                timeouts=sum(s['timeout'] for s in samples), errors=sum(s['status']=='ERROR' for s in samples),
                samples=len(samples), samples_during_prefill=len(during), poll_s=.2, timeout_s=3.0)


def ttft(client):
    import math
    frames = [f for f in client['frames'] if token_count(f['event'])]
    assert frames and client['status'] == 'VALID'
    value = frames[0]['elapsed_s']
    assert math.isfinite(value) and value > 0, 'missing/nonfinite measured TTFT'
    return value


def joined_request(client, text, ntokens=32):
    """Key -> trace Job serial -> REAL HTTP serial -> wrapped server completion line."""
    assert client['status'] == 'VALID', 'invalid client'
    traces = reuse.parse_traces(text)
    events = [t for t in traces if t.get('key') == client['key']]
    required = {}
    for kind in ('render', 'allocation', 'finish', 'final'):
        rows = [t for t in events if t['event'] == kind]
        assert len(rows) == 1, 'missing/duplicate ' + kind
        required[kind] = rows[0]
    assert len({t['serial'] for t in events if t.get('serial') is not None}) == 1, 'serial drift'
    render, alloc, finish = (required[k] for k in ('render', 'allocation', 'finish'))
    server = reuse.parse_server(text)[render['http_serial']]
    assert render['greedy'] and render['sampler_mode'] == 'greedy' and not render['heuristic']
    assert render['min_tokens'] == render['max_tokens'] == ntokens and render['loop_detect_window'] == 0
    assert (render['policy_pp'], render['policy_tail'], render['policy_near'], render['chunk'],
            render['policy_inforward']) == (4096, 12288, 2048, 2048, True)
    assert alloc['disk'] == 'off'
    assert finish['prompt_tokens'] == render['n'] == server['n'] == client['usage']['prompt_tokens']
    assert alloc['selected'] == finish['cached_tokens'] == server['n'] - server['new'] == 0, 'prompt not cold'
    assert client['usage'].get('prompt_tokens_details', {}).get('cached_tokens') == 0, 'frontend cache mismatch'
    assert finish['generated'] == client['usage']['completion_tokens'] == ntokens
    assert abs(finish['engine_prefill_s'] - server['engine_s']) <= 0.011, 'log/trace engine time mismatch'
    fw = [t for t in events if t['event'] == 'prefill_forward' and t['prompt_prefill']]
    assert fw and fw[0]['start'] == 0 and fw[-1]['end'] == render['n'] - 1
    assert all(a['end'] == b['start'] for a, b in zip(fw, fw[1:])), 'missing/duplicate prompt forward'
    assert not any(t['event'] == 'tail_fallback' for t in events), 'capture fallback'
    return dict(events=events, render=render, allocation=alloc, finish=finish, fw=fw, server=server)


def whole_window(row):
    # R825b trace has no window-ID event. With pinned source, a solo job, all forwards
    # pipelined and every interior capture crossed, a single window follows from
    # whole_window_ends/run_window. Requiring just pipeline:true would miss old windows.
    assert all(t['pipeline'] for t in row['fw']), 'pipeline:false in solo frontend request'
    end = row['fw'][-1]['end']
    captures = [t for t in row['events'] if t['event'] == 'tail_capture']
    assert captures, 'no capture evidence'
    for t in captures:
        assert t.get('stored'), 'unpublished capture'
        if t['position'] < row['fw'][-1]['start']:
            assert t.get('crossed') is True, 'interior window ended at a capture'
    # Every interior checkpoint must have an observed crossed publication, not just
    # one convenient true event. Daily policy: coarse32768, tail4096, near2048.
    render = row['render']
    for f in row['fw'][:-1]:
        pos = f['end']
        interval = (render['policy_near'] if pos >= end - 2 * render['chunk'] else
                    render['policy_pp'] if pos >= end - render['policy_tail'] else 32768)
        if pos % interval == 0:
            assert any(t['position'] == pos and t.get('crossed') is True for t in captures), 'uncrossed checkpoint'


def gap_metrics(frames, start, end):
    import math
    assert all(math.isfinite(x) for x in (start, end)) and end > start, 'bad prefill interval'
    times = [f['t'] for f in frames if token_count(f['event']) > 0]
    assert all(math.isfinite(t) for t in times) and all(b >= a for a, b in zip(times, times[1:])), 'bad SSE clocks'
    assert times and times[0] < start and times[-1] > end, 'decode did not bracket prefill'
    # Include the gap spanning either boundary: no excluding a full-window starvation.
    gaps = [b - a for a, b in zip(times, times[1:]) if a < end and b > start]
    tokens = sum(token_count(f['event']) for f in frames if start <= f['t'] <= end)
    assert gaps, 'missing inter-token gaps'
    return dict(longest_gap_s=max(gaps), tokens_during_prefill=tokens,
                prefill_window_s=end-start, tokens_per_s=tokens/(end-start),
                prefill_start_epoch=start, prefill_end_epoch=end,
                note='SSE delivery timestamps; grouped MTP tokens share a timestamp; boundary-spanning gaps included')


def frontend_check(prompts_path, client_path, text, candidate_arm=False):
    prompts = json.loads(Path(prompts_path).read_text())
    data = json.loads(Path(client_path).read_text())
    assert data['prompts'] == prompts, 'prompt artifact drift'
    cold = joined_request(data['cold50'], text)
    long = joined_request(data['long90'], text)
    assert abs(cold['render']['n'] - 50000) <= 512 and abs(long['render']['n'] - 90000) <= 512
    assert cold['render']['n'] == prompts['cold50']['n'] and long['render']['n'] == prompts['long90']['n']
    traces = reuse.parse_traces(text)
    keys = {data[name]['key'] for name in ('cold50','long90','decode','health90','arrival90','short200')}
    first = cold['render']['t']
    assert not any(t['event'] == 'render' and t['t'] >= first and t['key'] not in keys for t in traces), 'foreign frontend traffic'
    decode_renders = [t for t in traces if t['event'] == 'render' and t['key'] == data['decode']['key']]
    assert len(decode_renders) == 1 and decode_renders[0]['t'] < long['render']['t'], 'decode trace not first'
    assert decode_renders[0]['min_tokens'] == decode_renders[0]['max_tokens'] == 8192
    if candidate_arm:
        # Exclude foreign traffic throughout the solo cold50 (decode starts afterwards).
        renders = [t for t in reuse.parse_traces(text) if t['event'] == 'render']
        start, end = cold['render']['t'], cold['finish']['t']
        assert not any(start <= t['t'] <= end and t['key'] != data['cold50']['key'] for t in renders), 'foreign solo traffic'
        whole_window(cold)
    decoder = data['decode']
    assert decoder['status'] == 'VALID'
    # Trace allocation precedes first target chunk, final prompt forward follows its
    # GPU join. These actual server epochs bracket target prefill, including scheduling.
    start, end = long['allocation']['t'], long['fw'][-1]['t']
    assert decoder['dispatch_epoch'] < data['long90']['dispatch_epoch'], 'decode was not submitted first'
    assert sum(f['tokens'] > 0 and f['t'] < data['long90']['dispatch_epoch'] for f in decoder['frames']) >= 8, 'decode not active first'
    assert data['long90']['dispatch_epoch'] <= start < end <= next(f['t'] for f in data['long90']['frames'] if f['tokens']), 'server/client clock or prefill interval mismatch'
    assert end - start >= 0.8 * long['finish']['engine_prefill_s'], 'trace interval missing most prefill'
    assert decoder['end_epoch'] > end, 'decode completed before prefill end'
    metrics = gap_metrics(decoder['frames'], start, end)
    health = joined_request(data['health90'], text)
    arrival = joined_request(data['arrival90'], text)
    short = joined_request(data['short200'], text, 16)
    for name, row, target, tolerance in [('health90',health,90000,512), ('arrival90',arrival,90000,512), ('short200',short,200,20)]:
        assert row['render']['n'] == prompts[name]['n'] and abs(row['render']['n']-target) <= tolerance
    for name, row in [('health90',health), ('arrival90',arrival)]:
        assert not any(row['render']['t'] <= t['t'] <= row['fw'][-1]['t'] and t['key'] != data[name]['key']
                       for t in traces if t['event']=='render' and not (name == 'arrival90' and t['key'] == data['short200']['key'])), 'foreign solo/concurrency traffic'
    hs, he = health['allocation']['t'], health['fw'][-1]['t']
    assert data['health90']['dispatch_epoch'] <= hs < he <= next(f['t'] for f in data['health90']['frames'] if f['tokens'])
    assert he-hs >= .8 * health['finish']['engine_prefill_s'], 'health trace interval incomplete'
    health_result = health_metrics(data['health_samples'], hs, he)
    delay = data['short200']['dispatch_mono']-data['arrival90']['dispatch_mono']
    assert .95 <= delay <= 1.2, 'late arrival was not submitted at +1.0 s'
    assert arrival['allocation']['t'] < data['short200']['dispatch_epoch'] < arrival['fw'][-1]['t'], 'late request was not dispatched during cold prefill'
    arrival_result = dict(n=arrival['render']['n'], sha256=prompts['arrival90']['sha256'],
                          short_n=short['render']['n'], short_sha256=prompts['short200']['sha256'],
                          arrival_delay_s=delay, short_ttft_s=ttft(data['short200']))
    return dict(phase=data['phase'], cold50=dict(n=cold['render']['n'], cached=0,
                engine_s=cold['server']['engine_s'], trace_engine_s=cold['finish']['engine_prefill_s'],
                sha256=prompts['cold50']['sha256'], whole_window=candidate_arm),
                long90=dict(n=long['render']['n'], sha256=prompts['long90']['sha256'], **metrics),
                health90=dict(n=health['render']['n'], sha256=prompts['health90']['sha256'], **health_result),
                arrival90=arrival_result)


def compare_frontend(old, new):
    import math
    assert old['phase'] == 'old' and new['phase'] == 'candidate' and new['cold50']['whole_window']
    for name in ('cold50', 'long90', 'health90', 'arrival90'):
        assert old[name]['n'] == new[name]['n'] and old[name]['sha256'] == new[name]['sha256'], 'different prompts'
    a, b = old['cold50']['engine_s'], new['cold50']['engine_s']
    gap_a, gap_b = old['long90']['longest_gap_s'], new['long90']['longest_gap_s']
    assert all(math.isfinite(v) and v > 0 for v in (a,b,gap_a,gap_b)), 'nonfinite/zero measured gate'
    # Rich prints to hundredths: enforce the gate conservatively over rounding.
    assert b + 0.005 <= 0.95 * (a - 0.005), 'frontend cold50 engine saving <5% (rounded log uncertainty included)'
    assert gap_b <= 1.5 * gap_a, 'concurrent decode longest gap >1.5 x old daily'
    assert old['arrival90']['short_n'] == new['arrival90']['short_n'] and old['arrival90']['short_sha256'] == new['arrival90']['short_sha256'], 'different short prompts'
    ta, tb = old['arrival90']['short_ttft_s'], new['arrival90']['short_ttft_s']
    assert all(math.isfinite(v) and v > 0 for v in (ta,tb)), 'bad measured short TTFT'
    assert tb <= ta, 'late-arrival short TTFT exceeds old daily'
    # Old /health is evidence only: timeouts, non-200s and slow responses do NOT gate old.
    h = new['health90']
    assert math.isfinite(h['longest_latency_s']) and 0 <= h['longest_latency_s'] <= 1, 'candidate health latency >1 s'
    assert h['timeouts'] == 0 and h['errors'] == 0, 'candidate health timeout/error'
    return dict(cold50_ratio=b/a, longest_gap_ratio=gap_b/gap_a, short_ttft_ratio=tb/ta,
                old=old, candidate=new, verdict='PASS')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['pair','env','old-env','smoke-client','smoke-check','needles',
                                   'prepare','frontend-client','frontend-check','frontend-compare','evidence'])
    ap.add_argument('paths', nargs='*')
    a = ap.parse_args(); p = a.paths
    if a.mode == 'pair':
        assert Path(p[1]).read_text() == candidate(Path(p[0]).read_text()), 'launcher exceeds authorized delta'
        selectors(Path(p[1]).read_text())
    elif a.mode == 'env':
        check_env(Path(p[0]).read_text(), json.loads(Path(p[1]).read_text()), Path(p[2]).read_text())
    elif a.mode == 'old-env':
        check_old_env(json.loads(Path(p[0]).read_text()))
    elif a.mode == 'smoke-client':
        smoke_client(*p)
    elif a.mode == 'smoke-check':
        result = check_smoke(p[0], jsonl(p[1]), Path(p[2]).read_text(), p[3]); reuse.dump(p[4], result)
    elif a.mode == 'needles':
        check_needles(jsonl(p[0]), p[1])
    elif a.mode == 'prepare':
        prepare(*p)
    elif a.mode == 'frontend-client':
        frontend_client(*p)
    elif a.mode == 'frontend-check':
        reuse.dump(p[4], frontend_check(p[0], p[1], Path(p[2]).read_text(), p[3] == 'candidate'))
    elif a.mode == 'frontend-compare':
        reuse.dump(p[2], compare_frontend(json.loads(Path(p[0]).read_text()), json.loads(Path(p[1]).read_text())))
    else:
        verify_evidence()
    print('R825p ' + a.mode + ' PASS')


if __name__ == '__main__':
    main()
