import os
from tqdm import tqdm

FILE1 = "data/CLEANED_CORPUS.jsonl"
FILE2 = "data/final_data.jsonl"
OUTPUT = "/Volumes/KINGSTON/WEB_BOOKS_LITERARY.jsonl"

total_lines = 0

with open(OUTPUT, 'w', encoding='utf-8') as out:
    # -> append file 1
    print(f"appending {FILE1}...")
    with open(FILE1, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            out.write(line)
            total_lines += 1
    
    # -> append file 2
    print(f"appending {FILE2}...")
    with open(FILE2, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            out.write(line)
            total_lines += 1

# -> create file first
file_size_gb = os.path.getsize(OUTPUT) / (1024**3)
print(f"total lines: {total_lines:,}")
print(f"output size: {file_size_gb:.2f}GB")
print(f"output: {OUTPUT}")