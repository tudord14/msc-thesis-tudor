import json
import math
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# -> in this code we want to compute pplx for some models that are publicly available
# -> it's just a general comparison becuase the models differ so much in config
DATASETS = {
    "news": Path("transfer/MAI.jsonl"),
    "law": Path("transfer/new_law.jsonl"),
    "literary": Path("transfer/wikisource.jsonl"),
    "wikipedia": Path("transfer/wikipedia_full.jsonl"),
    "tech": Path("transfer/zonait.jsonl")
}
MODELS = [
    ("RoGPT2-medium", "readerbench/RoGPT2-medium"),
    ("RoGPT2-large", "readerbench/RoGPT2-large"),
    ("LLMic-3B", "faur-ai/LLMic"),
]
CONTEXT_LENGTH = 2048  
MAX_ARTICLES = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -> iterate just like for normal pplx
def iter_blocks(jsonl_path, tokenizer, block_size, max_articles=MAX_ARTICLES):
    buffer = []
    n = 0
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos = tokenizer.eos_token_id

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
            if bos is not None:
                ids = [bos] + ids
            if eos is not None:
                ids = ids + [eos]
            buffer += ids
            n += 1

            while len(buffer) >= block_size:
                block  = buffer[:block_size]
                buffer = buffer[block_size:]
                input_ids = torch.tensor(block, dtype=torch.long)
                labels    = input_ids.clone()
                labels[0] = -100
                yield input_ids, labels

# -> compute pplx
@torch.no_grad()
def compute_perplexity(model, tokenizer, jsonl_path, block_size):
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for input_ids, labels in iter_blocks(jsonl_path, tokenizer, block_size):
        input_ids = input_ids.unsqueeze(0).to(DEVICE)
        labels = labels.unsqueeze(0).to(DEVICE)
        try:
            out = model(input_ids=input_ids, labels=labels)
        except Exception as e:
            print(f"skip {e}")
            continue

        n_tokens = (labels != -100).sum().item()
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def main():
    available = {k: v for k, v in DATASETS.items() if v.exists()}
    missing   = [k for k in DATASETS if k not in available]
    if missing:
        print(f"missing datasets: {missing}")

    domains = list(available.keys())
    col = 14
    all_results = {}
    for label, hf_id in MODELS:
        print(f"loading {label} ({hf_id}) ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            model     = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map=DEVICE,
            )
        except Exception as e:
            print(f"fail to load {e}")
            all_results[label] = {d: None for d in domains}
            continue

        # -> determine safe block size for this model
        cfg = model.config
        max_pos = getattr(cfg, "max_position_embeddings",
                  getattr(cfg, "n_positions",
                  getattr(cfg, "n_ctx", 1024)))
        #block_size = min(CONTEXT_LENGTH, max_pos)
        MODEL_CONTEXT = {
    	"RoGPT2-medium": 1024,
    	"RoGPT2-large": 1024,
    	"LLMic-3B": 2048,
        }
        block_size = min(MODEL_CONTEXT.get(label, max_pos), max_pos)
        print(f"max_position_embeddings={max_pos}, using block_size={block_size}")

        all_results[label] = {}
        for domain, path in available.items():
            print(f"{domain} -> ", flush=True, end=" ")
            ppl = compute_perplexity(model, tokenizer, path, block_size)
            all_results[label][domain] = ppl
            print(f"ppl = {ppl:.2f}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    hdr = f"{'Model':<20} {'Params':<10}"
    for d in domains:
        hdr += f"{d:>{col}}"
    print(hdr)
    print("-" * len(hdr))
    sizes = {"RoGPT2-medium": "345M", "RoGPT2-large": "762M", "LLMic-3B": "3B"}
    for label, _ in MODELS:
        row = f"{label:<20} {sizes.get(label,'?'):<10}"
        for d in domains:
            v = all_results.get(label, {}).get(d)
            row += f"{v:>{col}.2f}" if v is not None else f"  {'N/A':>{col}}"
        print(row)

if __name__ == "__main__":
    main()
