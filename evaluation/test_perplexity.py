import json
import math
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

BASE = Path("")
MODELS_DIR = BASE / "models/models-good"
TOK_FILE = str(BASE / "ro_tokenizer_40k.json")

MODELS = {
    "llama_mha_baseline": MODELS_DIR / "llama_mha_baseline/final",
    #"llama_gqa": MODELS_DIR / "llama_gqa/final",
    #"falcon_mqa": MODELS_DIR / "falcon_mqa/final",
    #"mistral_sliding": MODELS_DIR / "mistral_sliding/final",
}

DATASETS = {
    "wikipedia": BASE / "transfer/wikipedia_full.jsonl",
    "police": BASE / "transfer/MAI.jsonl",
    "law": BASE / "transfer/new_law.jsonl",
    "basarabia": BASE / "transfer/basarabia.jsonl",
    "zonaIT": BASE / "transfer/zonait.jsonl",
}

CONTEXT_LENGTH = 2048  
MAX_ARTICLES = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
def load_tokenizer():
    tok = PreTrainedTokenizerFast(tokenizer_file=TOK_FILE)
    tok.bos_token_id = 0
    tok.eos_token_id = 1
    tok.pad_token_id = 3


def iter_blocks(jsonl_path, tokenizer, block_size=CONTEXT_LENGTH,
                max_articles=MAX_ARTICLES):
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


# -> actual pplx
@torch.no_grad()
def compute_perplexity(model, tokenizer, jsonl_path):
    model.eval()
    total_nll    = 0.0
    total_tokens = 0

    for input_ids, labels in iter_blocks(jsonl_path, tokenizer):
        input_ids = input_ids.unsqueeze(0).to(DEVICE)
        labels    = labels.unsqueeze(0).to(DEVICE)

        with torch.autocast(device_type=DEVICE, dtype=DTYPE):
            out = model(input_ids=input_ids, labels=labels)

        n_tokens      = (labels != -100).sum().item()  
        total_nll    += out.loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def main():
    tokenizer = load_tokenizer()
    domains = list(DATASETS.keys())

    print(f"\n{'model':<30}", end="")
    for d in domains:
        print(f"{d:>16}", end="")
    print("-" * (30 + 16 * len(domains)))
    # -> sweep
    for model_name, model_path in MODELS.items():
        print(f"\nloading {model_name} ...", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=DTYPE,       
            device_map=DEVICE,
        )
        model.eval()
        row = f"{model_name:<30}"
        for domain, path in DATASETS.items():
            if not path.exists():
                row += f"{'NOT FOUND':>16}"
                continue
            ppl = compute_perplexity(model, tokenizer, path)
            row += f"{ppl:>16.2f}"
            print(f"{domain}: PPL = {ppl:.2f}", flush=True)
 
        print(row)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
