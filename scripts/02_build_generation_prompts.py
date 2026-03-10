import argparse
import json
from pathlib import Path

import pandas as pd


def load_prompt(prompt_path: Path) -> str:
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", required=True, help="Path to human_abstracts.csv")
    parser.add_argument("--output", required=True, help="Output JSONL file for prompt jobs")
    parser.add_argument("--prompt-file", default="prompts/factual_prompt.txt")
    args = parser.parse_args()

    human_path = Path(args.human)
    out_path = Path(args.output)
    prompt_path = Path(args.prompt_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(human_path)
    required = {"source_id", "abstract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in human dataset: {sorted(missing)}")

    base_prompt = load_prompt(prompt_path)

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            source_id = str(row["source_id"])
            source_abs = str(row["abstract"]).strip()
            user_prompt = (
                f"{base_prompt}\n\n"
                f"Source abstract:\n{source_abs}\n\n"
                "Rewritten abstract:"
            )
            record = {
                "source_id": source_id,
                "prompt_type": "factual",
                "model_name": "llama-3.1",
                "messages": [
                    {"role": "system", "content": "You write high-quality academic abstracts."},
                    {"role": "user", "content": user_prompt},
                ],
                "generation_config": {
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "max_new_tokens": 450,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved prompt jobs to {out_path}")


if __name__ == "__main__":
    main()
