"""Start Tabby normally and record the loaded Flash-Next configuration."""

import functools
import json
import weakref


def emit(kind, evidence):
    print(f"REBASE167_{kind} {json.dumps(evidence, sort_keys=True)}", flush=True)


def install():
    from common import model
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.generator.job import Job

    load = model.load_model
    init_job = Job.__init__
    receive = Job.receive_sample

    @functools.wraps(load)
    async def load_and_record(*args, **kwargs):
        result = await load(*args, **kwargs)
        container = model.container
        selector = getattr(ext, "rope_fast_sincos_enabled", None)
        emit("RUNTIME", {
            "placement": {m.key: str(m.device) for m in container.model.modules},
            "rope": "fast_sincos" if selector is None or selector() else "range_reduced",
            "rope_source": "served native implementation" if selector is None else "native selector",
            "reasoning": [container.reasoning_start_token, container.reasoning_end_token],
        })
        return result

    @functools.wraps(init_job)
    def init_and_record(job, *args, **kwargs):
        init_job(job, *args, **kwargs)
        if job.name_policy.fast:
            ref = weakref.ref(job)

            @functools.wraps(receive)
            def receive_and_record(*args, **kwargs):
                current = ref()
                result = receive(current, *args, **kwargs)
                if result[0]:
                    emit("FAST", dict(current.fast_metrics))
                return result

            job.receive_sample = receive_and_record

    model.load_model = load_and_record
    Job.__init__ = init_and_record


if __name__ == "__main__":
    import main

    start = main.entrypoint_async

    async def start_audited():
        # This runs after main has configured its allocator and dependencies.
        install()
        await start()

    main.entrypoint_async = start_audited
    main.entrypoint()
