"""Capture stable greedy completion bytes for cross-image comparison; run on the GPU box."""
import argparse
import json
import os
from pathlib import Path
import urllib.request

PROMPTS = [
    "Explain why tides differ between ports. Give a precise technical explanation.",
    "Write a Python interval merging function with type hints, then explain edge cases.",
    "Explique en français la différence entre concurrence et parallélisme, avec exemples.",
    "用中文解释二分查找，并给出一个 Python 示例。",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="API base including /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    dest = Path(args.out_dir)
    dest.mkdir(parents=True, exist_ok=False)
    for i, prompt in enumerate(PROMPTS):
        body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": 256, "min_tokens": 256, "stream": False}
        headers = {"Content-Type": "application/json"}
        if os.environ.get("API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
        request = urllib.request.Request(args.url.rstrip("/") + "/chat/completions",
                                        json.dumps(body).encode(), headers)
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)
        choice = data["choices"][0]
        message = choice.get("message", {})
        captured = False
        for channel in ("content", "reasoning_content", "reasoning"):
            text = message.get(channel)
            if text is not None:
                (dest / f"{i}.{channel}.utf8").write_bytes(text.encode("utf-8"))
                captured |= bool(text)
        if not captured:
            raise RuntimeError(f"No output text in response {i}")
        metadata = {"finish_reason": choice.get("finish_reason"),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens")}
        (dest / f"{i}.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
