import argparse
import time
from typing import List, Optional

import requests
from dotenv import load_dotenv


load_dotenv()


def call_generate(base_url: str,
                  messages: List[dict],
                  temperature: float,
                  max_tokens: int,
                  repetition_penalty: float,
                  seed: Optional[int],
                  intervention: Optional[List[dict]],
                  stream: bool) -> str:
    url = base_url.rstrip("/") + "/generate"
    payload = {
        "prompt": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
        "stream": stream,
    }
    if intervention is not None:
        payload["intervention"] = intervention

    if stream:
        final_text = ""
        with requests.post(url, json=payload, stream=True, headers={"Accept": "text/event-stream"}) as r:
            r.raise_for_status()
            current_event = None
            for raw_line in r.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line[len("event: "):]
                elif line.startswith("data: "):
                    data = line[len("data: "):]
                    if current_event == "token":
                        print(data, end="", flush=True)
                        final_text += data
                    elif current_event == "done":
                        # server sends full text here; ensure final_text matches
                        final_text = data
                        print()
                        break
                    elif current_event == "error":
                        raise RuntimeError(data)
        return final_text
    else:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", "")


def main():
    parser = argparse.ArgumentParser(description="Test SAE steering via FastAPI app /generate endpoint")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000", help="Base URL of running app")
    parser.add_argument("--system", type=str, default="You are a helpful assistant who should follow the users requests.")
    parser.add_argument("--user", type=str, default="About 100 words, please give me some tourist information about Tokyo.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=324)
    parser.add_argument("--stream", action="store_true", help="Stream tokens using SSE")
    args = parser.parse_args()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.user},
    ]

    print("=== Baseline (no steering) ===")
    st = time.time()
    base_text = call_generate(
        base_url=args.base_url,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        intervention=None,
        stream=args.stream,
    )
    print("\n" + "-" * 100)
    if not args.stream:
        print(base_text)
        print("-" * 100)


    steered_text = call_generate(
        base_url=args.base_url,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        intervention=[{"feature_id": 58644, "value": 3.7}],
        stream=args.stream,
    )
    print("\n" + "-" * 100)
    if not args.stream:
        print(steered_text)
        print("-" * 100)


if __name__ == "__main__":
    main()


