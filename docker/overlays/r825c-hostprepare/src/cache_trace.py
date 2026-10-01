"""R823 CPU diagnostics: bounded arrays, existing page digests, keyed output hashes."""
from .cache.checkpoint_policy import interval as r823b_interval
import collections
import contextvars
import hashlib
import json
import os
import time
import uuid
ENABLED = os.environ.get('EXL3_CACHE_TRACE') == '1'
_SALT = os.urandom(32)
RUN = uuid.uuid4().hex
CONTEXT = contextvars.ContextVar('r823_context', default={})
REGISTRY = collections.OrderedDict()
HISTORY = collections.OrderedDict()
PEAK = 0


def digest(value):
    if value is None:
        return None
    if not isinstance(value, bytes):
        value = str(value).encode()
    return hashlib.blake2b(value, key=_SALT, digest_size=16).hexdigest()


def emit(event, **values):
    if ENABLED:
        print('[R823-cache] ' + json.dumps(dict(v=1, run=RUN, t=time.time(), event=event, **values),
                                         separators=(',', ':'), allow_nan=False), flush=True)


def request_context(request, data):
    if ENABLED:
        key = request.headers.get('x-r823-request')
        synthetic = bool(key and key.startswith('r823/') and len(key) <= 160)
        CONTEXT.set(dict(key=key if synthetic else digest(key or request.state.id),
                         conversation=digest(request.headers.get('x-r823-conversation')),
                         explicit=bool(request.headers.get('x-r823-conversation')),
                         roles=[m.role for m in getattr(data, 'messages', [])],http_serial=getattr(request.state,'serial',None)))


def encoded(ids, path):
    if ENABLED:
        emit('encode', **CONTEXT.get(), n=ids.shape[-1], path=path)


def lcp(a, b):
    if a is None or b is None:
        return None
    import numpy as np
    n = min(len(a), len(b))
    changed = np.flatnonzero(a[:n] != b[:n])
    return int(changed[0]) if len(changed) else n


def bounded():
    dropped = 0
    while len(REGISTRY) > 32 or sum(len(v.get('prompt', [])) + len(v.get('final', [])) for v in REGISTRY.values()) > 2097152:
        REGISTRY.popitem(last=False)
        dropped += 1
    if dropped:
        emit('registry_drop', entries=dropped, reason='bounded-registry')


def boundaries(ids, tokenizer):
    import numpy as np
    rows = []
    for p in np.flatnonzero(ids == tokenizer.single_id('<|im_start|>')):
        # Decode only the short role tag and whitelist it; no decoded content is emitted.
        import torch
        short = tokenizer.decode(torch.tensor([ids[int(p)+1:int(p)+5].tolist()]))[0]
        role = short.split('\\n')[0].split('\n')[0]
        rows.append(dict(position=int(p), role=role if role in ('system','user','assistant','tool') else 'unknown'))
    pat = tokenizer.encode('<tool_response>', add_bos=False, encode_special_tokens=True)[0].tolist()
    if pat:
        for p in np.flatnonzero(ids == pat[0]):
            if ids[int(p):int(p)+len(pat)].tolist() == pat:
                # tool messages are rendered inside a user wrapper in this template.
                rows.append(dict(position=int(p), role='tool'))
    rows.sort(key=lambda r:r['position'])
    rows = [r for i,r in enumerate(rows) if not (r['role']=='user' and i+1<len(rows) and rows[i+1]['role']=='tool')]
    for i,r in enumerate(rows):
        r['message_index'] = i
    return rows


def render(request_id, ids, tokenizer, params, sampler=None):
    if not ENABLED:
        return None
    a = ids[0].detach().cpu().numpy().copy()
    c = dict(CONTEXT.get())
    c.setdefault('key', digest(request_id))
    conv = c.get('conversation')
    heuristic = not c.get('explicit')
    if heuristic:
        candidates = [(lcp(a,v.get('prompt')),k) for k,v in REGISTRY.items()]
        conv = max(candidates, default=(0,digest(request_id)), key=lambda x:x[0] or 0)[1]
    prior = REGISTRY.get(conv,{})
    b = boundaries(a,tokenizer)
    p_lcp = lcp(a,prior.get('prompt'))
    changed = [i for i,r in enumerate(b) if p_lcp is not None and r['position'] <= p_lcp]
    steps=getattr(sampler,'steps',[])
    last=steps[-1] if steps else None
    is_greedy=type(last).__name__=='SS_Argmax' or (type(last).__name__=='SS_Fused' and last.mode==last.MODE_GREEDY)
    c.update(sampler_mode='greedy' if is_greedy else 'unknown',sampler_steps=[type(x).__name__ for x in steps],
             top_k=getattr(params,'top_k',None),top_p=getattr(params,'top_p',None),
             conversation=conv, heuristic=heuristic, n=len(a), prior=prior.get('key'),
             lcp_prompt=p_lcp, lcp_final=lcp(a,prior.get('final')), prior_final=prior.get('final_key'), boundaries=b,
             changed_message_index=changed[-1] if changed else None,
             raw_message_roles=c.pop('roles',[]), greedy=params.temperature == 0,
             min_tokens=params.min_tokens, max_tokens=params.max_tokens,
             loop_detect_window=params.loop_detect_window)
    REGISTRY[conv] = dict(prior,prompt=a,key=c['key'])
    REGISTRY.move_to_end(conv)
    bounded()
    return c


def owner(job):
    c = getattr(job,'_r823_trace',None) or {}
    return dict(key=c.get('key'),conversation=c.get('conversation'),serial=getattr(job,'serial_number',None))


def attach(job,record):
    if ENABLED and record is not None:
        job._r823_trace = record
        for seq in job.sequences:
            seq._r823_owner = job
        hashes = job.sequences[0].page_hashes
        generator = job.generator
        job._r823_pending_page_digests = hashes is None
        emit('render',**record,serial=job.serial_number,
             page_digests=None if hashes is None else [digest(h) for h in hashes],
             policy_pp=getattr(generator, "recurrent_checkpoint_interval_pp", None),
             policy_tail=getattr(generator, "recurrent_checkpoint_tail_pp", 0) if generator is not None else None,
             policy_inforward=os.environ.get("EXL3_RECURRENT_CHECKPOINT_INFORWARD", "0") == "1",
             policy_near=getattr(generator, "recurrent_checkpoint_interval", None),
             chunk=getattr(generator, "max_chunk_size", None))


def complete(job):
    if ENABLED and getattr(job,'_r823_trace',None):
        c = job._r823_trace
        a = job.sequences[0].sequence_ids.torch()[0].detach().cpu().numpy().copy()
        prior = REGISTRY.get(c['conversation'])
        if prior is not None:
            prior['final'] = a
            prior['final_key'] = c['key']
        bounded()
        emit('final',**owner(job),n=len(a),generated=len(a)-c['n'])


def occupancy(rc):
    if rc is None:
        return {}
    global PEAK
    PEAK = max(PEAK,rc.current_size)
    return dict(ram_bytes=rc.current_size,ram_max=rc.max_size,ram_entries=len(rc),ram_peak=PEAK,
                ram_metrics=dict(rc.metrics),
                pending=sum(not s._pm_ready.is_set() for s in rc.values() if hasattr(s,'_pm_ready')))


def before_allocation(seq,pt,rc):
    if not ENABLED:
        return None
    # Defence for callers that attach a trace before host preparation. None is
    # unknown, not an empty prompt. Publish the full list once at allocation.
    job = getattr(seq, '_r823_owner', None)
    if (getattr(job, '_r823_pending_page_digests', False)
            and seq.page_hashes is not None):
        emit('page_digests', **owner(job), page_digests=[digest(h) for h in seq.page_hashes])
        job._r823_pending_page_digests = False
    available = 0
    for h in seq.page_hashes:
        if pt.get_live_page(h) is None:
            break
        available += 1
    cap = len(seq.page_hashes) if seq.max_cached_pages is None else seq.max_cached_pages
    candidates = []
    for i,h in enumerate(seq.page_hashes):
        held = rc is not None and h in rc
        hist = HISTORY.get(h)
        if held or hist:
            candidates.append(dict(position=(i+1)*256,matching_hash=digest(h),present_RAM=held,
                                   anchored=i < available,capped=i >= cap,history=hist or 'held-before-tracing'))
    out = dict(pre_cap_kv_prefix=available*256,candidates=candidates,max_cached_pages=seq.max_cached_pages,
               kv_metrics=dict(pt.metrics),**occupancy(rc))
    emit('allocation_before',**owner(getattr(seq,'_r823_owner',None)),**out)
    return out


def after_allocation(seq,pt,rc,before,cached_pages,restore_limit):
    if not ENABLED:
        return
    from .cache.prefill_merge import metrics
    emit('allocation',**owner(getattr(seq,'_r823_owner',None)),kv_prefix=pt._r823_contiguous,
         selected=cached_pages*256,
         deepest_RAM=max((r['position'] for r in before['candidates'] if r['present_RAM'] and r['anchored'] and not r['capped']),default=0),
         disk='off' if pt.disk_tier is None else 'prepare_resume',restore_limit=restore_limit,
         kv_metrics=dict(pt.metrics),merge_async=dict(metrics),**occupancy(rc))


def save_reason(job,interval):
    seq = job.sequences[0]
    n = (getattr(job,'_r823_trace',None) or {}).get('n',len(seq.sequence_ids))-1
    if interval == 256 or seq.kv_position == n//256*256:
        return 'prompt_end'
    if seq.kv_position > n:
        return 'decode'
    return 'near_end' if seq.kv_position >= n-job.generator.max_chunk_size*2 else 'interior'


def remember(key,event):
    HISTORY[key] = event
    HISTORY.move_to_end(key)
    if len(HISTORY) > 8192:
        HISTORY.popitem(last=False)
        emit('history_drop',reason='bounded-history')


def saved(rc,key,state,existing,job,reason):
    if not ENABLED:
        return
    st = rc[key]
    remember(key,'saved')
    ready = getattr(st,'_pm_ready',None)
    emit('save',**owner(job),position=st.get('position',getattr(state,'position',None)),
         cached_base=getattr(job,'cached_pages',0)*256,reason=reason or 'other',
         effective_interval=r823b_interval(job.generator,st.get('position',state.position),(getattr(job,'_r823_trace',None) or {}).get('n',len(job.sequences[0].sequence_ids))-1) if job else None,
         digest=digest(key),inserted=not existing,bytes=st['checkpoint_size'],
         ready=ready is None or ready.is_set(),**occupancy(rc))


def loss(rc,key,st,reason,job):
    if not ENABLED:
        return
    remember(key,reason)
    emit('loss',**owner(job),digest=digest(key),position=st.get('position'),bytes=st['checkpoint_size'],reason=reason,
         anchored=rc.pagetable.is_resumable(key) if rc.pagetable else None,
         ram_bytes_after=sum(v['checkpoint_size'] for v in rc.values()))


def kv_loss(pt,page):
    if ENABLED:
        rc = pt.generator.recurrent_cache
        emit('kv_loss',**owner(getattr(pt,'_r823_owner',None)),digest=digest(page.phash),parent=digest(page.prev_hash),
             anchor=rc is not None and page.phash in rc,reason='complete-page-repurpose',kv_metrics=dict(pt.metrics))


def finished(job,result,finish):
    if ENABLED:
        emit('finish',**owner(job),prompt_tokens=result.get('prompt_tokens'),cached_tokens=result.get('cached_tokens'),
             generated=result.get('new_tokens'),queue_s=result.get('time_enqueued'),engine_prefill_s=result.get('time_prefill'),
             finish_reason=finish.get('finish_reason'),eos_reason=result.get('eos_reason'))
