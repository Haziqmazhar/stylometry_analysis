import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt(prompt_file: Path) -> str:
    with prompt_file.open("r", encoding="utf-8") as f:
        return f.read().strip()


def build_user_prompt(base_prompt: str, source_text: str) -> str:
    return f"{base_prompt}\n\nSource abstract:\n{source_text.strip()}\n\nRewritten abstract:"


def load_transformers_model(model_cfg: Dict) -> Tuple[object, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_cfg["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def generate_transformers(prompt: str, model_cfg: Dict, tokenizer: object, model: object) -> str:
    temperature = float(model_cfg.get("temperature", 0.4))
    top_p = float(model_cfg.get("top_p", 0.9))
    max_new_tokens = int(model_cfg.get("max_new_tokens", 450))

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def generate_ollama(prompt: str, model_cfg: Dict, host: str = "http://localhost:11434") -> str:
    import urllib.request

    payload = {
        "model": model_cfg["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": float(model_cfg.get("temperature", 0.4)),
            "top_p": float(model_cfg.get("top_p", 0.9)),
            "num_predict": int(model_cfg.get("max_new_tokens", 450)),
        },
    }
    req = urllib.request.Request(
        url=f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body.get("message", {}).get("content", "")).strip()


def maybe_load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def source_already_done(existing: pd.DataFrame, source_id: str, model_names: List[str]) -> bool:
    if existing.empty:
        return False
    rows = existing[existing["source_id"] == source_id]
    if rows.empty:
        return False
    have = set(rows["model_name"].astype(str))
    return all(m in have for m in model_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    gcfg = cfg["generation"]
    models = cfg["models"]

    human_path = Path(gcfg["human_input"])
    prompt_file = Path(gcfg["prompt_file"])
    out_csv = Path(gcfg["output_csv"])
    out_jsonl = Path(gcfg["output_jsonl"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    human = pd.read_csv(human_path)
    source_col = gcfg.get("source_column", "source_id")
    text_col = gcfg.get("text_column", "abstract")
    n = int(gcfg.get("sample_size", 300))
    resume = bool(gcfg.get("resume", True))
    strict_matrix = bool(gcfg.get("strict_matrix", True))
    prompt_type = gcfg.get("prompt_type", "factual")

    base_prompt = load_prompt(prompt_file)
    work = human[[source_col, text_col]].dropna().copy()
    work[source_col] = work[source_col].astype(str)
    work[text_col] = work[text_col].astype(str).str.strip()
    work = work[work[text_col].str.len() > 0].head(n)

    existing = maybe_load_existing(out_csv) if resume else pd.DataFrame()
    model_names = [m["name"] for m in models]

    all_rows: List[Dict] = []
    if not existing.empty:
        all_rows.extend(existing.to_dict(orient="records"))

    started = time.time()
    failed_pairs = 0
    generated_pairs = 0
    skipped_pairs = 0

    with out_jsonl.open("a", encoding="utf-8") as f_jsonl:
        existing_pairs = set()
        if not existing.empty:
            for _, r in existing.iterrows():
                existing_pairs.add((str(r["source_id"]), str(r["model_name"])))

        for model_cfg in models:
            backend = model_cfg.get("backend", "transformers").lower()
            model_name = model_cfg["name"]
            tok = mdl = None
            if backend == "transformers":
                print(f"Loading transformers model: {model_name} ({model_cfg['model_id']})")
                tok, mdl = load_transformers_model(model_cfg)

            for idx, (_, row) in enumerate(work.iterrows(), start=1):
                source_id = str(row[source_col])
                source_text = str(row[text_col])

                if resume and (source_id, model_name) in existing_pairs:
                    skipped_pairs += 1
                    continue

                user_prompt = build_user_prompt(base_prompt, source_text)
                try:
                    if backend == "transformers":
                        generated = generate_transformers(user_prompt, model_cfg, tok, mdl)
                    elif backend == "ollama":
                        generated = generate_ollama(user_prompt, model_cfg)
                    else:
                        raise ValueError(f"Unsupported backend: {backend}")
                except Exception as e:
                    failed_pairs += 1
                    print(f"[{idx}/{len(work)}] FAILED source_id={source_id} model={model_name} error={e}")
                    continue

                if not generated:
                    failed_pairs += 1
                    print(f"[{idx}/{len(work)}] EMPTY source_id={source_id} model={model_name}")
                    continue

                rec = {
                    "source_id": source_id,
                    "model_name": model_name,
                    "model_id": model_cfg["model_id"],
                    "backend": backend,
                    "prompt_type": prompt_type,
                    "generated_abstract": generated,
                    "temperature": float(model_cfg.get("temperature", 0.4)),
                    "top_p": float(model_cfg.get("top_p", 0.9)),
                }
                all_rows.append(rec)
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                generated_pairs += 1

                elapsed = time.time() - started
                print(
                    f"[{idx}/{len(work)}][{model_name}] generated_pairs={generated_pairs} "
                    f"failed_pairs={failed_pairs} skipped_pairs={skipped_pairs} elapsed={elapsed:.1f}s"
                )

            del tok, mdl

    out = pd.DataFrame(all_rows).drop_duplicates(subset=["source_id", "model_name"], keep="last")
    if strict_matrix and not out.empty:
        wanted = set(model_names)
        source_to_models = out.groupby("source_id")["model_name"].apply(lambda s: set(s.astype(str)))
        keep_sources = [sid for sid, mods in source_to_models.items() if wanted.issubset(mods)]
        out = out[out["source_id"].isin(keep_sources)].copy()

    out.to_csv(out_csv, index=False)
    total_elapsed = time.time() - started
    h = int(total_elapsed // 3600)
    m = int((total_elapsed % 3600) // 60)
    s = total_elapsed % 60
    print(f"Saved {len(out)} rows to {out_csv}")
    print(f"Runtime: {h:02d}:{m:02d}:{s:05.2f}")


if __name__ == "__main__":
    main()
