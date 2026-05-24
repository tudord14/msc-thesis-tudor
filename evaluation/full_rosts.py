import os
import sys
import time
import signal
import subprocess
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


"""
RO-STS kinda complete pipeline  
1. Fine-tune BF16 models
2. Save complete CausalLM checkpoint
3. Convert to GGUF FP16
4. Quantize to Q8_0, Q5_K_M, Q4_K_M
5. Evaluate all variants via llama-server embeddings
"""

THESIS_DIR = Path.home() / "MSC-THESIS"
MODELS_DIR = THESIS_DIR / "models/models-good"
TOKENIZER = THESIS_DIR / "ro_tokenizer_40k.json"
FINETUNED_DIR = THESIS_DIR / "models/rosts_finetuned"
TRAIN_FILE = THESIS_DIR / "text-similarity/RO-STS.train.tsv"
DEV_FILE = THESIS_DIR / "text-similarity/RO-STS.dev.tsv"
TEST_FILE = THESIS_DIR / "text-similarity/RO-STS.test.tsv"
LLAMA_CPP_DIR = Path.home() / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
LLAMA_QUANTIZE = LLAMA_CPP_DIR / "build/bin/llama-quantize"
LLAMA_SERVER = LLAMA_CPP_DIR / "build/bin/llama-server"
BF16_MODELS = {
    "Llama-MHA": MODELS_DIR / "llama_mha_baseline/final",
    "Llama-GQA": MODELS_DIR / "llama_gqa/final",
    "Mistral-SWA": MODELS_DIR / "mistral_sliding/final",
    "Falcon-MQA": MODELS_DIR / "falcon_mqa/final",
}

MAX_LENGTH = 128
BATCH_SIZE = 32
LR_START = 1e-4
LR_END = 4e-6
PATIENCE = 5
MAX_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SERVER_PORT = 9090
QUANT_TYPES = ["Q8_0", "Q5_K_M", "Q4_K_M"]

_server_proc = None
def load_tsv(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            score_raw = parts[0].strip().replace(",", ".")
            text1 = " ".join(parts[1].split())
            text2 = " ".join("\t".join(parts[2:]).split())
            try:
                score = float(score_raw)
            except ValueError:
                continue
            rows.append((score, text1, text2))
    df = pd.DataFrame(rows, columns=["score", "text1", "text2"])
    if df.empty:
        raise RuntimeError(f"No valid rows in {path}")
    max_sc = df["score"].max()
    df["gold01"] = df["score"] / (5.0 if max_sc > 1.0 else 1.0)
    return df


class STSDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc1 = self.tokenizer(row["text1"], max_length=self.max_len,
                              truncation=True, padding="max_length",
                              return_tensors="pt")
        enc2 = self.tokenizer(row["text2"], max_length=self.max_len,
                              truncation=True, padding="max_length",
                              return_tensors="pt")
        return {
            "input_ids_1":      enc1["input_ids"].squeeze(0),
            "attention_mask_1": enc1["attention_mask"].squeeze(0),
            "input_ids_2":      enc2["input_ids"].squeeze(0),
            "attention_mask_2": enc2["attention_mask"].squeeze(0),
            "label":            torch.tensor(row["gold01"], dtype=torch.float32),
        }


# -> wrap backbone, model intact
class STSModel(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.full_model = full_model
        if hasattr(full_model, "model"):
            self._backbone_attr = "model"        
        elif hasattr(full_model, "transformer"):
            self._backbone_attr = "transformer"
        else:
            raise ValueError("Unknown model architecture")

    @property
    def backbone(self):
        return getattr(self.full_model, self._backbone_attr)

    def mean_pool(self, hidden, mask):
        mask_exp = mask.unsqueeze(-1).float()
        summed = (hidden * mask_exp).sum(dim=1)
        counts = mask_exp.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        pooled = self.mean_pool(out.hidden_states[-1], attention_mask)
        return torch.nn.functional.normalize(pooled, dim=-1)

    def forward(self, input_ids_1, attention_mask_1,
                input_ids_2, attention_mask_2):
        u = self.encode(input_ids_1, attention_mask_1)
        v = self.encode(input_ids_2, attention_mask_2)
        return (u * v).sum(dim=-1)


# -> finetune the model 
def finetune_and_save(model_name, model_path, train_df, dev_df, test_df):
    print(f"\n  Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), tokenizer_file=str(TOKENIZER), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -> wrap full backbone
    full_model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16
    )
    model = STSModel(full_model).to(DEVICE)
    loss_fn = nn.MSELoss()
    # -> opt and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=1.0 / np.e,
        patience=2, min_lr=LR_END
    )
    # -> load data
    tr_loader = DataLoader(STSDataset(train_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(STSDataset(dev_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    te_loader = DataLoader(STSDataset(test_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    best_pearson = -1.0
    best_state = None
    patience_cnt = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(tr_loader, desc=f"  Epoch {epoch:02d}", leave=False):
            pred = model(
                batch["input_ids_1"].to(DEVICE),
                batch["attention_mask_1"].to(DEVICE),
                batch["input_ids_2"].to(DEVICE),
                batch["attention_mask_2"].to(DEVICE)
            )
            loss = loss_fn(pred, batch["label"].to(DEVICE))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for batch in dev_loader:
                p = model(
                    batch["input_ids_1"].to(DEVICE),
                    batch["attention_mask_1"].to(DEVICE),
                    batch["input_ids_2"].to(DEVICE),
                    batch["attention_mask_2"].to(DEVICE)
                )
                preds.extend(p.float().cpu().numpy().tolist())
                golds.extend(batch["label"].numpy().tolist())

        pr, _ = pearsonr(golds, preds)
        sr, _ = spearmanr(golds, preds)
        mse = mean_squared_error(golds, preds)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:02d}  loss={total_loss/len(tr_loader):.5f}  "
              f"dev_r={pr:.4f}  dev_ρ={sr:.4f}  mse={mse:.5f}  lr={lr_now:.2e}")

        scheduler.step(pr)
        if pr > best_pearson + 1e-4:
            best_pearson = pr
            best_state = {k: v.detach().cpu()
                          for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("  Early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in te_loader:
            p = model(
                batch["input_ids_1"].to(DEVICE),
                batch["attention_mask_1"].to(DEVICE),
                batch["input_ids_2"].to(DEVICE),
                batch["attention_mask_2"].to(DEVICE)
            )
            preds.extend(p.float().cpu().numpy().tolist())
            golds.extend(batch["label"].numpy().tolist())

    pr, _ = pearsonr(golds, preds)
    sr, _ = spearmanr(golds, preds)
    mse = mean_squared_error(golds, preds)
    print(f"\n  ── BF16 TEST ── Pearson={pr:.4f}  Spearman={sr:.4f}  MSE={mse:.5f}")

    bf16_metrics = {"pearson": round(float(pr), 4),
                    "spearman": round(float(sr), 4),
                    "mse": round(float(mse), 5)}

    # -> save complete causal lm
    ft_path = FINETUNED_DIR / model_name
    ft_path.mkdir(parents=True, exist_ok=True)
    print(f"saving causallm to {ft_path}")
    model.full_model.save_pretrained(ft_path)
    tokenizer.save_pretrained(ft_path)

    del model, full_model
    torch.cuda.empty_cache()

    return bf16_metrics, ft_path


# -> convert to gguf
def convert_to_gguf(hf_path, gguf_out):
    cmd = [sys.executable, str(CONVERT_SCRIPT),
           "--outfile", str(gguf_out), "--outtype", "f16", str(hf_path)]
    print(f"converting -> {gguf_out.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"convert error:\n{result.stderr[-500:]}")
        raise RuntimeError("gguf conversion fail")


def quantize_gguf(src, dst, quant_type):
    cmd = [str(LLAMA_QUANTIZE), str(src), str(dst), quant_type]
    print(f"  Quantizing {quant_type} -> {dst.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  QUANTIZE ERROR:\n{result.stderr[-300:]}")
        raise RuntimeError(f"Quantization {quant_type} failed")


# -> help of gemini
def start_server(gguf_path):
    global _server_proc
    stop_server()
    cmd = [
        str(LLAMA_SERVER),
        "-m", str(gguf_path),
        "--embeddings",
        "--pooling", "mean",
        "--port", str(SERVER_PORT),
        "--ctx-size", "128",
        "--threads", "16",
        "--batch-size", "512",
        "--no-webui",
        "--log-disable",
    ]
    print(f"  Starting server for {gguf_path.name}...")
    _server_proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    for i in range(60):
        time.sleep(1)
        try:
            r = requests.get(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2)
            if r.status_code == 200:
                print(f"  Server ready ({i+1}s)")
                return True
        except Exception:
            pass

    try:
        _, stderr = _server_proc.communicate(timeout=2)
        print(f"  SERVER ERROR:\n{stderr.decode()[-500:]}")
    except Exception:
        pass
    stop_server()
    return False


def stop_server():
    global _server_proc
    if _server_proc is not None:
        try:
            os.killpg(os.getpgid(_server_proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            _server_proc.wait(timeout=5)
        except Exception:
            pass
        _server_proc = None
        time.sleep(2)

# -> get embeddings
def get_embedding(text):
    for url, payload in [
        (f"http://127.0.0.1:{SERVER_PORT}/v1/embeddings",
         {"input": text, "encoding_format": "float"}),
        (f"http://127.0.0.1:{SERVER_PORT}/embedding",
         {"content": text}),
    ]:
        try:
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "data" in data and len(data["data"]) > 0:
                return np.array(data["data"][0]["embedding"], dtype=np.float32)
            if "embedding" in data:
                return np.array(data["embedding"], dtype=np.float32)
        except Exception:
            continue
    raise RuntimeError("emb failed on both endpoints")


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def evaluate_gguf(gguf_path, test_df):
    if not start_server(gguf_path):
        return None
    try:
        preds = []
        golds = test_df["gold01"].tolist()
        for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                           desc=f"  Eval {gguf_path.name}", leave=False):
            try:
                u = get_embedding(row["text1"])
                v = get_embedding(row["text2"])
                preds.append(cosine_sim(u, v))
            except Exception as e:
                print(f"\n  WARNING: {e}")
                preds.append(0.0)

        pr, _ = pearsonr(golds, preds)
        sr, _ = spearmanr(golds, preds)
        mse = mean_squared_error(golds, preds)
        print(f"  Pearson={pr:.4f}  Spearman={sr:.4f}  MSE={mse:.5f}")
        return {"pearson": round(float(pr), 4),
                "spearman": round(float(sr), 4),
                "mse": round(float(mse), 5)}
    finally:
        stop_server()


def main():
    for f in [TRAIN_FILE, DEV_FILE, TEST_FILE]:
        if not f.exists():
            print(f"ERROR: {f} not found")
            sys.exit(1)

    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_tsv(TRAIN_FILE)
    dev_df = load_tsv(DEV_FILE)
    test_df = load_tsv(TEST_FILE)
    print(f"train={len(train_df)}  dev={len(dev_df)}  test={len(test_df)}")
    print(f"device: {DEVICE}")
    all_results = {}
    for name, path in BF16_MODELS.items():
        print(f"model -> {name}")

        if not path.exists():
            print("skip")
            continue

        model_results = {}

        try:
            # -> fine tune and save
            bf16_metrics, ft_path = finetune_and_save(
                name, path, train_df, dev_df, test_df
            )
            model_results["BF16"] = bf16_metrics

            # ─> convert to gguf
            slug = name.lower().replace("-", "_")
            gguf_fp16 = ft_path / f"{slug}_fp16.gguf"
            if not gguf_fp16.exists():
                convert_to_gguf(ft_path, gguf_fp16)

            # -> eval fp16
            m = evaluate_gguf(gguf_fp16, test_df)
            if m:
                model_results["FP16"] = m
                print(f"FP16={m['pearson']:.4f} S={m['spearman']:.4f}")

            # -> quantize and eval
            for q in QUANT_TYPES:
                q_path = ft_path / f"{slug}_{q}.gguf"
                if not q_path.exists():
                    quantize_gguf(gguf_fp16, q_path, q)

                m = evaluate_gguf(q_path, test_df)
                if m:
                    model_results[q] = m
                    print(f"{q} -> P={m['pearson']:.4f} S={m['spearman']:.4f}")
            all_results[name] = model_results

        except Exception as e:
            print(f"err: {e}")
            import traceback
            traceback.print_exc()
            stop_server()

    stop_server()

    # -> prints
    baselines = [
        ("BERT-base-ro",  0.8159, 0.8086),
        ("RoBERT-large",  0.8376, 0.8315),
        ("RoGPT2-medium", 0.8316, 0.8225),
        ("RoGPT2-large",  0.8346, 0.8264),
    ]

    print(f"\n{'Model':<20} {'Variant':<10} {'Pearson':>10} {'Spearman':>10}")
    for bname, pr, sr in baselines:
        print(f"{bname:<20} {'—':<10} {pr:>10.4f} {sr:>10.4f}")
    for name, variants in all_results.items():
        for var, m in variants.items():
            print(f"{name:<20} {var:<10} {m['pearson']:>10.4f} {m['spearman']:>10.4f}")
 

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
        stop_server()
