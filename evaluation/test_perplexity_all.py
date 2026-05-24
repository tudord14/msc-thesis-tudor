import json
import math
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


# -> configuration for evaluating pplx on ALL models
BASE = Path("/home/tdiac/MSC-THESIS")
MODELS_DIR = BASE / "models/models-good"
QUANT_DIR = BASE / "models/quantized"
TOK_FILE = str(BASE / "ro_tokenizer_40k.json")

# -> modelf folders
MODELS = {
    "llama_mha_baseline": MODELS_DIR / "llama_mha_baseline/final",
#    "llama_gqa": MODELS_DIR / "llama_gqa/final",
#    "mistral_sliding": MODELS_DIR / "mistral_sliding/final",
#    "falcon_mqa": MODELS_DIR / "falcon_mqa/final",
}
# -> quantizations
GGUF_VARIANTS = [
    ("FP16", "fp16"),
    ("Q4_K_M", "Q4_K_M"),
    ("Q5_K_M", "Q5_K_M"),
    ("Q8_0", "Q8_0"),
]

DATASETS = {
    "police": BASE / "transfer/MAI.jsonl",
    "law": BASE / "transfer/new_law.jsonl",
    "basarabia": BASE / "transfer/basarabia.jsonl",
    "wikipedia": BASE / "transfer/wikipedia_full.jsonl",
    "tech_reviews": BASE / "transfer/zonait.jsonl"
}
CONTEXT_LENGTH = 2048
MAX_ARTICLES   = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
def load_tokenizer():
    tok = PreTrainedTokenizerFast(tokenizer_file=TOK_FILE)
    tok.bos_token_id = 0
    tok.eos_token_id = 1
    tok.pad_token_id = 3
    return tok

# -> block iterator we keep it same for all
def iter_blocks(jsonl_path, tokenizer, block_size=CONTEXT_LENGTH, max_articles=MAX_ARTICLES):
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    buffer = []
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_articles and n >= max_articles:
                break
            try:
                text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                continue
            if not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            buffer += [bos] + ids + [eos]
            n += 1
            while len(buffer) >= block_size:
                block  = buffer[:block_size]
                buffer = buffer[block_size:]
                input_ids = torch.tensor(block, dtype=torch.long)
                labels    = input_ids.clone()
                labels[0] = -100
                yield input_ids, labels

# -> compute actual pplx
@torch.no_grad()
def compute_perplexity_hf(model, tokenizer, jsonl_path):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for input_ids, labels in iter_blocks(jsonl_path, tokenizer):
        input_ids = input_ids.unsqueeze(0).to(DEVICE)
        labels = labels.unsqueeze(0).to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=DTYPE):
            out = model(input_ids=input_ids, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)

# -> this is the tricky part, compute for quantized
def compute_perplexity_gguf(gguf_path, tokenizer, jsonl_path):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("no llama-cpp")
        return None
    try:
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=CONTEXT_LENGTH,
            n_threads=16,
            n_gpu_layers=0,
            logits_all=True,
            verbose=False,
        )
    except Exception as e:
        print(f"error {e}")
        return None

    total_nll = 0.0
    total_tokens = 0

    for input_ids, labels in iter_blocks(jsonl_path, tokenizer):
        tokens = input_ids.tolist()
        llm.reset()
        llm.eval(tokens)
        # -> scores[i] = raw logits -> apply log-softmax to get log-probs
        raw = np.array(llm.scores[:len(tokens)], dtype=np.float64)
        raw -= raw.max(axis=1, keepdims=True) 
        log_probs_all = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

        for i in range(len(tokens) - 1):
            if labels[i + 1].item() == -100:
                continue
            target   = tokens[i + 1]
            log_prob = float(log_probs_all[i][target])
            total_nll    += -log_prob
            total_tokens += 1

    del llm
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def main():
    tokenizer = load_tokenizer()
    available = {k: v for k, v in DATASETS.items() if v.exists()}
    if not available:
        print("no datasets")
        return
    missing = [k for k in DATASETS if k not in available]
    if missing:
        print(f"missing datasets: {missing}")

    domains = list(available.keys())
    col = 14
    all_results = {}
    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"\skip {model_name} BF16 -> not found")
            continue

        print(f"\n{model_name} (BF16) ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=DTYPE, device_map=DEVICE,
        )
        model.eval()
        key = (model_name, "BF16")
        all_results[key] = {}
        for domain, path in available.items():
            print(f"{domain}...", flush=True, end=" ")
            ppl = compute_perplexity_hf(model, tokenizer, path)
            all_results[key][domain] = ppl
            print(f"ppl = {ppl:.2f}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    for model_name in MODELS:
        model_dir = QUANT_DIR / model_name
        if not model_dir.exists():
            print(f"\skip {model_name} GGUF - directory not found")
            continue

        for label, suffix in GGUF_VARIANTS:
            gguf = model_dir / f"{model_name}_{suffix}.gguf"
            if not gguf.exists():
                print(f"\skip {gguf.name} - not found")
                continue

            print(f"\n{model_name} [{label}]", flush=True)
            key = (model_name, label)
            all_results[key] = {}

            for domain, path in available.items():
                print(f"{domain}->", flush=True, end=" ")
                ppl = compute_perplexity_gguf(gguf, tokenizer, path)
                all_results[key][domain] = ppl
                if ppl is not None:
                    print(f"ppl = {ppl:.2f}")
                else:
                    print("failed")

    hdr = f"{'Model':<28} {'Variant':<10}"
    for d in domains:
        hdr += f"  {d:>{col}}"
    print(hdr)

    variant_order = ["BF16"] + [label for label, _ in GGUF_VARIANTS]
    for model_name in MODELS:
        for variant in variant_order:
            key = (model_name, variant)
            if key not in all_results:
                continue
            row = f"{model_name:<28} {variant:<10}"
            for d in domains:
                v = all_results[key].get(d)
                row += f"{v:>{col}.2f}" if v is not None else f"  {'n/a':>{col}}"
            print(row)
        print()


if __name__ == "__main__":
    main()
