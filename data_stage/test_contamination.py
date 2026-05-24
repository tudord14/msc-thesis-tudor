import json
import re
import sys
from pathlib import Path
from datasketch import MinHash, MinHashLSH


# -> we employ this in order to test the contamination degree of our saved datasets
# -> this way we make sure our 
TRAIN_FILE = Path("preprocessing/WEB_BOOKS_LITERARY.jsonl")
EVAL_FILES = {
     "basarabia": Path("transfer/basarabia.jsonl"),
     "zonaIT": Path("transfer/zonait.jsonl")
#    "agonia": Path("transfer/agonia.jsonl"),
#    "police": Path("transfer/MAI.jsonl"),
#    "law": Path("transfer/new_law.jsonl"),
#    "protv1": Path("transfer/pro_tv1.jsonl"),
#    "news_full": Path("news_full.jsonl"),
#    "wikisource": Path("wikisource.jsonl"),
}

# -> some parameters 
THRESHOLD = 0.85    # -> Jaccard similarity threshold to call a "match"
NUM_PERM = 128    # -> MinHash permutations 
NGRAM_SIZE = 5      # -> character n-grams 
MAX_TRAIN = None  
REPORT_EVERY = 100_000

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_minhash(text: str, num_perm: int = NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    text = normalize(text)
    for i in range(len(text) - NGRAM_SIZE + 1):
        m.update(text[i:i+NGRAM_SIZE].encode("utf-8"))
    return m

def read_texts(jsonl_path: Path, max_docs=None):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            try:
                yield json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                continue

# -> hash all eval docs upfront -> small -> fit RAM
eval_hashes = {}   # {dataset_name: [(doc_index, MinHash)]}
for name, path in EVAL_FILES.items():
    if not path.exists():
        print(f"warning: {path} not found -> skip")
        continue
    hashes = []
    for i, text in enumerate(read_texts(path)):
        hashes.append((i, make_minhash(text)))
    eval_hashes[name] = hashes
    print(f"{name}: {len(hashes)} docs hashed")

if not eval_hashes:
    print("smth wrong")
    sys.exit(1)

# -> we index the eval docs and query with train docs
lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
key_to_dataset = {}   # -> maps "name_i" -> dataset name
for name, hashes in eval_hashes.items():
    for i, mh in hashes:
        key = f"{name}_{i}"
        lsh.insert(key, mh)
        key_to_dataset[key] = name

total_eval = sum(len(v) for v in eval_hashes.values())
print(f"-> inndex contains {total_eval} eval docs total")

# -> stream train corpus
print(f"\n-> {TRAIN_FILE}")
print(f"sim threshold: {THRESHOLD} -- n-gram size: {NGRAM_SIZE}")
print(f"this may take a while....\n")

# -> contamination counters per eval dataset
hit_counts = {name: 0 for name in eval_hashes}
total_train  = 0

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if MAX_TRAIN and total_train >= MAX_TRAIN:
            break
        total_train += 1

        if total_train % REPORT_EVERY == 0:
            print(f"processed {total_train:,} training docs...", flush=True)
        try:
            text = json.loads(line)["text"]
        except (json.JSONDecodeError, KeyError):
            continue

        mh = make_minhash(text)
        results = lsh.query(mh)

        for key in results:
            dataset = key_to_dataset[key]
            hit_counts[dataset] += 1
            lsh.remove(key)
            del key_to_dataset[key]

        if not key_to_dataset:
            print("all eval docs accounted for — stopping early.")
            break

# -> report
print(f"training docs scanned: {total_train:,}")
print(f"sim threshold:{THRESHOLD}")
print(f"{'Dataset':<20} {'Eval Docs':>10} {'Matches':>10} {'Overlap %':>10}")

any_contamination = False
for name, hashes in eval_hashes.items():
    n_eval = len(hashes)
    n_hits = hit_counts[name]
    pct = 100 * n_hits / n_eval if n_eval > 0 else 0
    flag = "contaminated" if pct > 5 else " ✓ clean"
    print(f"{name:<20} {n_eval:>10} {n_hits:>10} {pct:>9.1f}%{flag}")
    if pct > 5:
        any_contamination = True

if any_contamination:
    print("warning: some eval sets have significant overlap with training data")
else:
    print("all eval sets appear clean —> safe to use for pplx")
