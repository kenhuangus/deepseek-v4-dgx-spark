#!/usr/bin/env python3
"""Smoke test script for DeepSeek-V4.1-Flash on dgx-spark-llm port 8888."""
import argparse
import json
import sys
import urllib.error
import urllib.request


def test_models(base_url: str) -> bool:
    print(f"Testing {base_url}/models ...")
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=10) as resp:
            data = json.load(resp)
            print(f"Status: {resp.status}")
            print(json.dumps(data, indent=2))
            return True
    except Exception as e:
        print(f"Error fetching models: {e}")
        return False


def test_chat(base_url: str, model_name: str = "deepseek-v4.1-flash") -> bool:
    print(f"\nTesting {base_url}/chat/completions with model={model_name} ...")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Explain what an operating system kernel is in one clear sentence."}
        ],
        "temperature": 1.0,
        "max_tokens": 128,
        "chat_template_kwargs": {"reasoning_effort": 75},
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
            choice = data["choices"][0]["message"]
            content = choice.get("content", "")
            reasoning = choice.get("reasoning_content", "")
            timings = data.get("timings", {})
            print(f"Status: {resp.status}")
            print(f"Reasoning:\n{reasoning}\n")
            print(f"Generated text:\n{content}\n")
            print(f"Prompt Speed: {timings.get('prompt_per_second', 'N/A')} t/s")
            print(f"Decode Speed: {timings.get('predicted_per_second', 'N/A')} t/s")
            return True
    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code}")
        print(e.read().decode("utf-8", "replace"))
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek-V4.1-Flash on dgx-spark-llm")
    parser.add_argument("--base-url", default="http://localhost:8888/v1", help="Base API URL")
    parser.add_argument("--model", default="deepseek-v4.1-flash", help="Model name or alias")
    args = parser.parse_args()

    if not test_models(args.base_url):
        sys.exit(1)
    if not test_chat(args.base_url, args.model):
        print("Falling back to alias 'deepseek'...")
        test_chat(args.base_url, "deepseek")


if __name__ == "__main__":
    main()
