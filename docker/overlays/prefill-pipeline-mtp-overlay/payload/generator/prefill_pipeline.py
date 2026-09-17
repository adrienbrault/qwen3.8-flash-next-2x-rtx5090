"""Opt-in, two-device Qwen prefill lookahead. Never handles a decode forward.

One outstanding stage-0 chunk at most. Both devices are quiescent before returning
to Job's draft/commit continuation. Caller streams are reused: new stream-local
allocator arenas would waste the served configuration's small memory margin.
"""
from concurrent.futures import ThreadPoolExecutor

import torch

from ..cache.recurrent_util import advance_recurrent_states
from ..model.model_ls import _LSDeviceContext
from ..util.device_copy import needs_bounce
from ..util.prefill_nosync import enabled as _prefill_nosync_enabled


def window_ends(start, prompt_end, checkpoint, limit):
    """Only original full 2048-token chunks, ending at the first checkpoint.

    checkpoint is evaluated at the *completed* chunk position. A checkpoint must
    never observe stage 0 one chunk ahead of stage 1. Leave partial tails to Job.
    """
    ends = []
    last_page = prompt_end // 256 * 256
    while len(ends) < limit:
        end = min((start + 2048) // 256 * 256, prompt_end)
        if start < last_page <= end:
            end = last_page
        if end - start != 2048:
            break
        ends.append(end)
        if checkpoint(end):
            break
        start = end
    return ends


def run_window(job, results, runtime, count):
    """The real driver, also exercised with a CPU-only fake stage executor."""
    job._ls_prefill_pipeline = runtime
    try:
        for _ in range(count):
            job.prefill(results)  # target, then existing draft, pages and progress
            if job.generator.recurrent_cache is not None:
                job.maybe_stash_recurrent(job.generator.recurrent_cache)
        assert runtime.pending is None
    finally:
        try:
            runtime.drain()
        finally:
            del job._ls_prefill_pipeline


class _PeerCopy:
    """R417's checked consumer-stream UVA handoff, without its batch splitter."""
    def __init__(self, devices):
        import ctypes
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.copy = self.driver.cuMemcpyAsync
        self.copy.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                              ctypes.c_size_t, ctypes.c_void_p]
        self.copy.restype = ctypes.c_int
        get_context = self.driver.cuCtxGetCurrent
        get_context.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        get_context.restype = ctypes.c_int
        enable_peer = self.driver.cuCtxEnablePeerAccess
        enable_peer.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        enable_peer.restype = ctypes.c_int
        # Register peer mappings with PyTorch's allocator, including VMM mappings.
        bootstrap = torch.zeros(1, device = devices[0]).to(devices[1])
        bootstrap.to(devices[0])
        contexts = []
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.current_stream(device)
                context = ctypes.c_void_p()
                self.check(get_context(ctypes.byref(context)))
                if not context.value:
                    raise RuntimeError("Prefill pipeline: missing CUDA context")
                contexts.append(context)
        for i, device in enumerate(devices):
            with torch.cuda.device(device):
                result = enable_peer(contexts[1 - i], 0)
                if result != 704:  # CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
                    self.check(result)

    @staticmethod
    def check(result):
        if result:
            raise RuntimeError(f"Prefill pipeline CUDA driver error: {result}")

    def __call__(self, source, device, stream):
        assert source.is_cuda and source.is_contiguous()
        assert source.device != device and stream.device == device
        dest = torch.empty(source.shape, dtype = source.dtype, device = device)
        self.check(self.copy(dest.data_ptr(), source.data_ptr(),
                            source.numel() * source.element_size(), stream.cuda_stream))
        source.record_stream(stream)
        return dest


def _layout(model):
    if (type(model).__name__ != "Qwen4ExpModel" or model.loaded_tp or
            getattr(model, "component", "text") != "text" or
            model.config.hc_mult != 4 or model.config.hidden_size != 2560 or
            getattr(model.config, "moe_cpu_hosts", {}) or
            model.config.infer_params.no_reconstruct):
        return None
    modules = model.fwd_modules
    if len({id(m) for m, _, _ in modules}) != len(modules):
        return None
    prefix = 0
    while prefix < len(modules) and modules[prefix][0].device.type == "cpu":
        if type(modules[prefix][0]).__name__ != "Embedding":
            return None
        prefix += 1
    gpu = modules[prefix:]
    if not gpu or any(m.device.type != "cuda" for m, _, _ in gpu):
        return None
    d0 = gpu[0][0].device
    boundary = next((i for i, (m, _, _) in enumerate(gpu) if m.device != d0), None)
    if boundary is None:
        return None
    d1 = gpu[boundary][0].device
    if model.output_device != d1 or any(m.device != d1 for m, _, _ in gpu[boundary:]):
        return None
    stages = (gpu[:boundary], gpu[boundary:])
    # Cache validation of the immutable module/submodule load layout.
    signature = tuple((id(m), m.device, inst, idx) for m, inst, idx in modules)
    if getattr(model, "_ls_prefill_checked", None) != signature:
        seen = {}
        for stage, device in zip(stages, (d0, d1)):
            for module, _, _ in stage:
                for sub in module:
                    if sub.device != device or seen.get(id(sub), device) != device:
                        return None
                    seen[id(sub)] = device
                    if type(sub).__name__ == "BlockSparseMLP" and (
                            sub.cpu_offload or sub.cpu_split_first is not None):
                        return None
                if type(module).__name__ == "PLELayer" and device != d0:
                    return None
        if model.last_kv_module_idx_instance not in [(idx, inst) for _, inst, idx in stages[1]]:
            return None
        model._ls_prefill_checked = signature
    if (not torch.cuda.can_device_access_peer(d0, d1) or
            not torch.cuda.can_device_access_peer(d1, d0) or
            needs_bounce(d0, d1) or needs_bounce(d1, d0)):
        return None
    return modules[:prefix], stages, (d0, d1), signature


class _Runtime:
    def __init__(self, job, ends, layout, executor, peer_copy):
        self.job, self.ends = job, ends
        self.model = job.generator.model
        self.mtp = job.generator.mtp_draft
        self.prefix, self.stages, self.devices, _ = layout
        self.streams = tuple(torch.cuda.current_stream(d) for d in self.devices)
        self.executor, self.peer_copy = executor, peer_copy
        self.index = 0
        self.pending = None

    def _modules(self, x, params, modules):
        ctx = _LSDeviceContext()
        try:
            for module, instance, idx in modules:
                params["layer_instance"] = instance
                last = (idx, instance) == self.model.last_kv_module_idx_instance
                if not self.mtp:
                    params["prefill"] = last
                if self.mtp and module.caps.get("logits_output"):
                    x = x[..., -1:, :].contiguous()
                ctx.enter(module.device)
                x = module.forward(module.prepare_for_device(x, params), params)
                if not self.mtp and last:
                    break
        finally:
            ctx.restore()
        return x

    def _stage0(self, ids, params):
        # inference mode and current CUDA device/stream are thread-local.
        with torch.inference_mode(), torch.cuda.device(self.devices[0]), torch.cuda.stream(self.streams[0]):
            params["dev_cache"] = {}
            params["pinned_staging"] = False
            if self.mtp:
                params["last_tokens_only"] = 1
            x = self.model.prepare_inputs(ids, params)
            # Own host metadata; do not race the persistent per-tensor device cache.
            for key in ("block_table", "cache_seqlens", "positions", "recurrent_slots"):
                if params.get(key) is not None:
                    params[key] = params[key].clone().pin_memory()
            # A_i has already run PLE before A_(i+1) is submitted. Its CPU ID history
            # is current even though B_i and the logical position commit are pending.
            for module in self.model._get_prefetch_layers:
                module.prefetch(x, params)
            x = self._modules(x, params, self.prefix)
            if x.device.type == "cpu" and not x.is_pinned():
                x = x.pin_memory()
            x = self._modules(x, params, self.stages[0])
            x = x.clone(memory_format = torch.contiguous_format)
            done = torch.cuda.Event()
            done.record(self.streams[0])  # producer device, before any next chunk
            return ids, params, x, done

    def drain(self):
        error = None
        try:
            if self.pending is not None:
                self.pending.result()  # finish host issue before joining GPU work
        except BaseException as exc:
            error = exc
        # Attempt BOTH joins even if the worker or the first device raised.
        for device, stream in zip(self.devices, self.streams):
            try:
                with torch.cuda.device(device):
                    stream.synchronize()
            except BaseException as exc:
                if error is None:
                    error = exc
        self.pending = None
        if error is not None:
            raise error

    def forward(self, ids, params):
        """Called only at Job.prefill's target dispatch, never Model.forward/decode."""
        with torch.inference_mode():
            start = int(params["cache_seqlens"][0])
            assert ids.shape == (1, 2048) and start + 2048 == self.ends[self.index]
            raw = params.copy()
            if self.pending is None:
                descriptor = self._stage0(ids, params)
            else:
                descriptor = self.pending.result()
                self.pending = None
                assert torch.equal(ids, descriptor[0])
                assert int(descriptor[1]["cache_seqlens"][0]) == start
            _, p, source, ready = descriptor
            try:
                with torch.cuda.device(self.devices[1]), torch.cuda.stream(self.streams[1]):
                    self.streams[1].wait_event(ready)
                    x = self.peer_copy(source, self.devices[1], self.streams[1])
                if self.index + 1 < len(self.ends):
                    next_ids = self.job.sequences[0].sequence_ids.torch_slice(
                        self.ends[self.index], self.ends[self.index + 1])
                    raw["cache_seqlens"] = torch.tensor([self.ends[self.index]], dtype = torch.int32)
                    self.pending = self.executor.submit(self._stage0, next_ids, raw)
                # Host readbacks in MoE require independent issue of A_(i+1) and B_i.
                # The worker touches device 0 only; device 1 remains on this thread.
                with torch.cuda.device(self.devices[1]), torch.cuda.stream(self.streams[1]):
                    self._modules(x, p, self.stages[1])
                if self.pending is not None:
                    self.pending.result()
                # MTP may use either device and the same device-local scratch. Complete
                # BOTH stages before Job resumes draft_i, then commit_i. No lookahead
                # crosses a checkpoint, so checkpoint snapshots remain coherent.
                if _prefill_nosync_enabled(p):
                    from ..util.prefill_nosync import join_stages
                    join_stages(self.devices, self.streams)
                else:
                    for device, stream in zip(self.devices, self.streams):
                        with torch.cuda.device(device):
                            stream.synchronize()
            except BaseException:
                self.drain()
                raise  # never retry a partially executed chunk
            if not self.mtp:
                p.pop("prefill", None)
            params.update(p)
            advance_recurrent_states(ids, params, self.model)
            self.index += 1


def prefill_job(job, results):
    """Return False before any state mutation when this window is unsupported."""
    gen = job.generator
    if (type(gen.model).__name__ != "Qwen4ExpModel" or
            gen.max_chunk_size != 2048 or len(job.sequences) != 1 or job.embeddings or
            job.alt_rope_freqs is not None or job.is_requeued or
            job.recurrent_state is None or job.recurrent_state.exported or
            (gen.draft_model is not None and
             (not gen.mtp_draft or type(gen.draft_model).__name__ != "Qwen4ExpMTPModel"))):
        return False
    seq = job.sequences[0]
    start, prompt_end = seq.kv_position, len(seq.sequence_ids) - 1
    # Generator stamps time_first_token before its first draft/decode forward;
    # Job.prefill never sets it. Unlike new_tokens, it survives a decode rewind.
    # The token counter is not a prefill/decode phase marker.
    if (job.time_first_token is not None or seq.prefill_complete or
            start % 256 or job.recurrent_state.position != start):
        return False
    # Other active jobs bound the prefill burst, even if still prefilling. Decode
    # assembly, speculative depth selection and the full target pass remain unchanged.
    limit = 2 if len(gen.active_jobs) > 1 else (prompt_end - start) // 2048
    def checkpoint(end):
        interval = (gen.recurrent_checkpoint_interval if end >= prompt_end - 4096
                    else gen.recurrent_checkpoint_interval_pp)
        return (end - job.cached_pages * 256) % interval == 0
    ends = window_ends(start, prompt_end, checkpoint, limit)
    if len(ends) < 2:
        return False
    pages = seq.allocated_pages[start // 256:ends[-1] // 256]
    if (len(pages) != (ends[-1] - start) // 256 or
            any(p.kv_position != 0 or p.ref_count != 1 for p in pages) or
            len({p.page_index for p in pages}) != len(pages)):
        return False  # retain prefix reuse/replay/copy-on-write handling in Job
    layout = _layout(gen.model)
    if layout is None:
        return False
    devices = layout[2]
    # Extra FP32 boundary/activation retention plus host-thread library workspace.
    # Existing scratch is reused on the caller streams, not cloned into new arenas.
    if any(torch.cuda.mem_get_info(d)[0] < 320 * 1024**2 for d in devices):
        return False
    setup = getattr(gen.model, "_ls_prefill_setup", None)
    if setup is None or setup[0] != layout[3]:
        if setup is not None:
            setup[1].shutdown(wait = True)
        peer_copy = _PeerCopy(devices)  # all setup failures precede live chunk work
        executor = ThreadPoolExecutor(max_workers = 1, thread_name_prefix = "ls_prefill")
        setup = gen.model._ls_prefill_setup = (layout[3], executor, peer_copy)
        gen.model.ls_prefill_pipeline_layout = {
            "devices": [str(d) for d in devices],
            "stages": [[(m.key, inst, idx) for m, inst, idx in stage] for stage in layout[1]],
        }
        print(f" -- LS prefill pipeline: {gen.model.ls_prefill_pipeline_layout}")
    runtime = _Runtime(job, ends, layout, setup[1], setup[2])
    run_window(job, results, runtime, len(ends))
    return True
