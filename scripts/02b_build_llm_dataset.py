import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="Raw generation outputs JSONL")
    parser.add_argument("--output-csv", required=True, help="Output llm_abstracts.csv")
    parser.add_argument("--model-name", default="llama-3.1")
    parser.add_argument("--prompt-type", default="factual")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            source_id = str(rec.get("source_id", "")).strip()
            text = str(rec.get("generated_abstract", rec.get("output_text", ""))).strip()
            if not source_id or not text:
                continue
            rows.append(
                {
                    "source_id": source_id,
                    "model_name": args.model_name,
                    "prompt_type": args.prompt_type,
                    "generated_abstract": text,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["source_id"])
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
