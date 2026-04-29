import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import yaml
from pandas.errors import EmptyDataError


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
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def load_existing_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def model_slug(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_name)


def per_model_paths(base_csv: Path, base_jsonl: Path, model_name: str) -> Tuple[Path, Path]:
    slug = model_slug(model_name)
    csv_dir = base_csv.parent / "per_model"
    jsonl_dir = base_jsonl.parent / "per_model"
    csv_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir / f"{slug}.csv", jsonl_dir / f"{slug}.jsonl"


def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def persist_csv(rows: List[Dict], out_csv: Path, strict_matrix: bool, model_names: List[str]) -> None:
    if not rows:
        return
    out = pd.DataFrame(rows).drop_duplicates(subset=["source_id", "model_name"], keep="last")
    if strict_matrix and not out.empty:
        wanted = set(model_names)
        source_to_models = out.groupby("source_id")["model_name"].apply(lambda s: set(s.astype(str)))
        keep_sources = [sid for sid, mods in source_to_models.items() if wanted.issubset(mods)]
        out = out[out["source_id"].isin(keep_sources)].copy()
    out.to_csv(out_csv, index=False)


def append_json_record(handle, record: Dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def append_generation_record(
    record: Dict,
    all_rows: List[Dict],
    global_jsonl_handle,
    model_jsonl_handle,
    out_csv: Path,
    strict_matrix: bool,
    model_names: List[str],
    model_csv: Path,
    per_model_rows: List[Dict],
) -> List[Dict]:
    all_rows.append(record)
    append_json_record(global_jsonl_handle, record)
    append_json_record(model_jsonl_handle, record)
    persist_csv(all_rows, out_csv, strict_matrix, model_names)
    per_model_rows.append(record)
    persist_csv(per_model_rows, model_csv, False, [record["model_name"]])
    return per_model_rows


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
    require_columns(human, [source_col, text_col], str(human_path))
    n = int(gcfg.get("sample_size", 300))
    resume = bool(gcfg.get("resume", True))
    strict_matrix = bool(gcfg.get("strict_matrix", True))
    prompt_type = gcfg.get("prompt_type", "factual")

    base_prompt = load_prompt(prompt_file)
    work = human[[source_col, text_col]].dropna().copy()
    work[source_col] = work[source_col].astype(str)
    work[text_col] = work[text_col].astype(str).str.strip()
    work = work[work[text_col].str.len() > 0].head(n)

    if resume:
        existing_csv = maybe_load_existing(out_csv)
        existing_jsonl = load_existing_jsonl(out_jsonl)
        existing = pd.concat([existing_csv, existing_jsonl], ignore_index=True) if not existing_csv.empty or not existing_jsonl.empty else pd.DataFrame()
        if not existing.empty:
            require_columns(existing, ["source_id", "model_name"], f"{out_csv} / {out_jsonl}")
            existing = existing.drop_duplicates(subset=["source_id", "model_name"], keep="last")
    else:
        existing = pd.DataFrame()
    model_names = [m["name"] for m in models]
    per_model_files = {m["name"]: per_model_paths(out_csv, out_jsonl, m["name"]) for m in models}

    all_rows: List[Dict] = []
    if not existing.empty:
        all_rows.extend(existing.to_dict(orient="records"))

    started = time.time()
    failed_pairs = 0
    generated_pairs = 0
    skipped_pairs = 0

    try:
        with out_jsonl.open("a", encoding="utf-8") as f_jsonl:
            existing_pairs: Set[Tuple[str, str]] = set()
            if not existing.empty:
                for _, r in existing.iterrows():
                    existing_pairs.add((str(r["source_id"]), str(r["model_name"])))

            for model_cfg in models:
                backend = model_cfg.get("backend", "transformers").lower()
                model_name = model_cfg["name"]
                model_csv, model_jsonl = per_model_files[model_name]
                tok = mdl = None
                per_model_existing = pd.concat(
                    [maybe_load_existing(model_csv), load_existing_jsonl(model_jsonl)],
                    ignore_index=True,
                )
                if not per_model_existing.empty:
                    require_columns(per_model_existing, ["source_id", "model_name"], f"{model_csv} / {model_jsonl}")
                    per_model_existing = per_model_existing.drop_duplicates(
                        subset=["source_id", "model_name"], keep="last"
                    )
                per_model_rows = per_model_existing.to_dict(orient="records") if not per_model_existing.empty else []
                try:
                    if backend == "transformers":
                        print(f"Loading transformers model: {model_name} ({model_cfg['model_id']})")
                        tok, mdl = load_transformers_model(model_cfg)
                except Exception as e:
                    failed_pairs += len(work)
                    print(f"FAILED model load model={model_name} error={e}")
                    persist_csv(all_rows, out_csv, strict_matrix, model_names)
                    continue

                with model_jsonl.open("a", encoding="utf-8") as f_model_jsonl:
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
                            persist_csv(all_rows, out_csv, strict_matrix, model_names)
                            continue

                        if not generated:
                            failed_pairs += 1
                            print(f"[{idx}/{len(work)}] EMPTY source_id={source_id} model={model_name}")
                            persist_csv(all_rows, out_csv, strict_matrix, model_names)
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
                        per_model_rows = append_generation_record(
                            record=rec,
                            all_rows=all_rows,
                            global_jsonl_handle=f_jsonl,
                            model_jsonl_handle=f_model_jsonl,
                            out_csv=out_csv,
                            strict_matrix=strict_matrix,
                            model_names=model_names,
                            model_csv=model_csv,
                            per_model_rows=per_model_rows,
                        )
                        existing_pairs.add((source_id, model_name))
                        generated_pairs += 1

                        elapsed = time.time() - started
                        print(
                            f"[{idx}/{len(work)}][{model_name}] generated_pairs={generated_pairs} "
                            f"failed_pairs={failed_pairs} skipped_pairs={skipped_pairs} elapsed={elapsed:.1f}s"
                        )

                del tok, mdl
    finally:
        persist_csv(all_rows, out_csv, strict_matrix, model_names)

    out = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    total_elapsed = time.time() - started
    h = int(total_elapsed // 3600)
    m = int((total_elapsed % 3600) // 60)
    s = total_elapsed % 60
    print(f"Saved {len(out)} rows to {out_csv}")
    print(f"Runtime: {h:02d}:{m:02d}:{s:05.2f}")


if __name__ == "__main__":
    main()
