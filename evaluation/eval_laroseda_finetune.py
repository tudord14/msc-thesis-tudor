import json
import random
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE = Path("/home/tdiac/MSC-THESIS")
MODELS_DIR = BASE / "models/models-good"
TOKENIZER = BASE / "ro_tokenizer_40k.json"

MODELS = {
    "Llama-MHA": MODELS_DIR / "llama_mha_baseline/final",
    "Llama-GQA": MODELS_DIR / "llama_gqa/final",
    "Mistral-SWA": MODELS_DIR / "mistral_sliding/final",
    "Falcon-MQA": MODELS_DIR / "falcon_mqa/final",
}
LEARNING_RATE = 1e-4
EPOCHS = 5
BATCH_SIZE = 32
MAX_LENGTH = 256
WARMUP_STEPS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -> gemini my friend did this
def download_file(url, dest_path):
    import subprocess, urllib.request
    print(f"  Downloading {dest_path.name} ...")
    r = subprocess.run(["wget", "-q", "-O", str(dest_path), url], capture_output=True)
    if r.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"    OK via wget ({dest_path.stat().st_size // 1024} KB)")
        return
    r = subprocess.run(["curl", "-sL", "-o", str(dest_path), url], capture_output=True)
    if r.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"    OK via curl ({dest_path.stat().st_size // 1024} KB)")
        return
    urllib.request.urlretrieve(url, dest_path)
    print(f"    OK via urllib ({dest_path.stat().st_size // 1024} KB)")

# -> load the data up
def load_laroseda():
    cache_dir = BASE / "data" / "laroseda"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pos_path = cache_dir / "positive_reviews.json"
    neg_path = cache_dir / "negative_reviews.json"
    base_url = "https://raw.githubusercontent.com/ancatache/LaRoSeDa/main/data"
    for path, url in [(pos_path, f"{base_url}/positive_reviews.json"), (neg_path, f"{base_url}/negative_reviews.json")]:
        if path.exists() and path.stat().st_size > 1000:
            print(f"cached: {path.name}")
        else:
            download_file(url, path)

    def parse(json_path, label):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [{"title": (item.get("title") or "").strip(),
                 "content": (item.get("content") or "").strip(),
                 "label": label}
                for item in data.get("reviews", [])]

    pos = parse(pos_path, 1)
    neg = parse(neg_path, 0)
    print(f"-> {len(pos)} positive, {len(neg)} negative")
    random.seed(42)
    random.shuffle(pos); random.shuffle(neg)
    train = pos[:6000] + neg[:6000]
    test  = pos[6000:] + neg[6000:]
    random.shuffle(train); random.shuffle(test)
    print(f"train: {len(train)}, test: {len(test)}")
    return train, test

# -> actual dataset
class LaRoSeDaDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        t, c = s.get("title",""), s.get("content","")
        text = f"{t}. {c}" if t and c else (t or c)
        enc  = self.tokenizer(text, max_length=self.max_len, truncation=True,
                              padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(s["label"], dtype=torch.long)}

# -> classifier
class SentimentClassifier(nn.Module):
    def __init__(self, backbone, hidden_size, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(self.dropout(pooled))

# -> actual train run
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(ids, mask), lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total += loss.item()
    return total / len(loader)

# -> eval
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y_pred.extend(model(ids, mask).argmax(-1).cpu().numpy().tolist())
            y_true.extend(batch["label"].numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, p, r, f1


def run_model(name, path, train_samples, test_samples):
    print(f"model: {name}")
    if not path.exists():
        print("err skip")
        return None
    # -> tokenize
    tokenizer = AutoTokenizer.from_pretrained(str(path), tokenizer_file=str(TOKENIZER), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -> load data
    train_loader = DataLoader(LaRoSeDaDataset(train_samples, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader  = DataLoader(LaRoSeDaDataset(test_samples,  tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # -> get backbone and train
    backbone = AutoModel.from_pretrained(str(path), torch_dtype=torch.bfloat16)
    model = SentimentClassifier(backbone, backbone.config.hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, len(train_loader) * EPOCHS)

    best = {"acc": 0, "f1": 0, "prec": 0, "rec": 0}
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler)
        acc, prec, rec, f1 = evaluate(model, test_loader)
        print(f"epoch {epoch}; loss={loss:.4f}; acc={acc:.4f}; f1={f1:.4f}")
        if acc > best["acc"]:
            best = {"acc": acc, "f1": f1, "prec": prec, "rec": rec}
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                print("early")
                break

    del model, backbone
    torch.cuda.empty_cache()
    print(f"best acc={best['acc']:.4f} f1={best['f1']:.4f}")
    return {"model": name, **best}


# -> copilot helped
# -> main
def main():
    print("LaRoSeDa — Fine-tuning with Classification Head")
    print(f"Matching RoGPT2 paper methodology (Niculescu et al., 2021)")
    print(f"Device: {DEVICE}\n")

    train_samples, test_samples = load_laroseda()

    results = []
    for name, path in MODELS.items():
        r = run_model(name, path, train_samples, test_samples)
        if r: results.append(r)

    print("\n\n" + "=" * 55)
    print("LAROSEDA — FINE-TUNED BINARY SENTIMENT CLASSIFICATION")
    print("=" * 55)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1 Macro':>10}")
    print("-" * 42)
    for n, a, f in [("RoBERT-large*",0.9820,0.9819),("RoGPT2-large*",0.9806,0.9807),
                    ("RoGPT2-medium*",0.9803,0.9804),("RoGPT2-base*",0.9789,0.9788)]:
        print(f"{n:<20} {a:>10.4f} {f:>10.4f}")
    print("-" * 42)
    for r in results:
        print(f"{r['model']:<20} {r['acc']:>10.4f} {r['f1']:>10.4f}")
    print("=" * 55)

    out = BASE / "laroseda_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
