# tokcount-r1: the requeue token-count fix on the served image

`tabbyapi:stack-r3-rows32-tokcount` is `tabbyapi:stack-r3-rows32` plus [`fix.patch`](fix.patch), one line of `exllamav3/generator/job.py`: `Job.prepare_for_requeue` carries `rq_new_tokens + new_tokens` instead of `new_tokens`, so a generation's count accumulates across requeued segments. Without it, `usage.completion_tokens` and the logged tokens per second report a generation longer than about 4,096 tokens as its last one or two segments ([GOTCHAS 4](../../../docs/GOTCHAS.md)). The same fix was a `sed` in `docker/Dockerfile.tabbyapi`; the chain from `Dockerfile.tabbyapi-qsa-cid` on installs ExLlamaV3 afresh and did not carry it.

```sh
docker build -f Dockerfile.box --build-arg BASE=tabbyapi:stack-r3-rows32 -t tabbyapi:stack-r3-rows32-tokcount .
```

The build applies the patch at fuzz 0 after a dry run (`--forward`, so a second apply fails), removes `__pycache__`, and runs [`landing_tokcount.py`](landing_tokcount.py) with every `EXL3_*` key stripped: it imports the module on CPU, checks the new expression by AST and the absence of the old line, and checks that `rq_new_tokens` has one reader besides the requeue path, the finish report. Python only, about 30 s, no GPU.

[`tokcount_check.py`](tokcount_check.py) is the probe of [R737 and R747](../../../bench/results/r747-tokcount.md): streamed generations forced to N tokens (`min_tokens`, `loop_detect_window: 0`), whose `usage` and server-log counts must both equal N.
