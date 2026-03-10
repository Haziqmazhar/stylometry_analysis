import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict, timeout: int = 180) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def load_existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = str(rec.get("source_id", "")).strip()
                if sid:
                    ids.add(sid)
            except json.JSONDecodeError:
                continue
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="Prompt jobs JSONL from 02_build_generation_prompts.py")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with generated abstracts")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-records", type=int, default=0, help="0 means all")
    parser.add_argument("--resume", action="store_true", help="Skip source_id already present in output file")
    args = parser.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    existing_ids = load_existing_ids(out_path) if args.resume else set()
    if existing_ids:
        print(f"Resume mode: {len(existing_ids)} existing records will be skipped.")

    with in_path.open("r", encoding="utf-8") as fin:
        jobs = [json.loads(line) for line in fin if line.strip()]

    if args.max_records > 0:
        jobs = jobs[: args.max_records]

    total = len(jobs)
    done = 0
    skipped = 0
    started_at = time.time()

    with out_path.open("a", encoding="utf-8") as fout:
        for idx, job in enumerate(jobs, start=1):
            source_id = str(job.get("source_id", "")).strip()
            if not source_id:
                skipped += 1
                continue
            if source_id in existing_ids:
                skipped += 1
                continue

            messages = job.get("messages", [])
            gen = job.get("generation_config", {})
            prompt_type = str(job.get("prompt_type", "factual"))

            payload = {
                "model": args.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": float(gen.get("temperature", 0.4)),
                    "top_p": float(gen.get("top_p", 0.9)),
                    "num_predict": int(gen.get("max_new_tokens", 450)),
                },
            }

            last_error = None
            response = None
            for attempt in range(1, args.retries + 1):
                try:
                    response = post_json(f"{args.host}/api/chat", payload)
                    last_error = None
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
                    last_error = e
                    time.sleep(min(2 * attempt, 10))

            if last_error is not None or response is None:
                print(f"[{idx}/{total}] FAILED source_id={source_id}: {last_error}")
                continue

            content = (
                response.get("message", {}).get("content", "")
                if isinstance(response, dict)
                else ""
            )
            content = str(content).strip()
            if not content:
                print(f"[{idx}/{total}] EMPTY source_id={source_id}")
                continue

            out_rec = {
                "source_id": source_id,
                "model_name": args.model,
                "prompt_type": prompt_type,
                "generated_abstract": content,
                "temperature": float(gen.get("temperature", 0.4)),
                "top_p": float(gen.get("top_p", 0.9)),
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            done += 1
            elapsed = time.time() - started_at
            print(
                f"[{idx}/{total}] generated={done} skipped={skipped} "
                f"source_id={source_id} elapsed={elapsed:.1f}s"
            )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

            if idx % 25 == 0 or idx == total:
                print(f"Progress: {idx}/{total} processed | written={done} | skipped={skipped}")

    total_elapsed = time.time() - started_at
    h = int(total_elapsed // 3600)
    m = int((total_elapsed % 3600) // 60)
    s = total_elapsed % 60
    print(f"Completed. Written={done}, skipped={skipped}, total_input={total}")
    print(f"Total runtime: {h:02d}:{m:02d}:{s:05.2f}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
