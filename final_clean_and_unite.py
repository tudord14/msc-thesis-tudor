import json
import os
import re
from pathlib import Path
from tqdm import tqdm

# -> final cleaning script before training tokenizer
BOOKS_CLEANED_FILE = "data/books_cleaned.jsonl"
CLEANED_CORPUS_FILE = "data/CLEANED_CORPUS.jsonl"
AGONIA_FILE = "data/agonia.jsonl"
OUTPUT_JSONL = "data/final_data.jsonl"
FINAL_OUTPUT = "data/WEB_BOOKS_LITERARY.jsonl"

def clean_final_text(text):
    # -> remove control characters but keep newlines
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = ''.join(ch for ch in text if not (ord(ch) < 32 and ch not in '\n\t\r'))
    
    # -> remove angle brackets with garbage
    text = re.sub(r'<[^>]*>', '', text)
    
    # -> remove stray unicode markers and corrupted sequences
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # -> clean up weird characters inside words (not at boundaries)
    # -> keep Romanian special chars but remove random symbols
    text = re.sub(r'([a-zăâîșț])[\^\*\`~\|\\]+([a-zăâîșț])', r'\1\2', text, flags=re.IGNORECASE)
    
    # -> remove multiple spaces, tabs
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    
    # -> remove excessive newlines (more than 2 in a row)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    # -> filter out metadata/garbage lines
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        # -> count actual letters (Romanian + English)
        letter_count = sum(1 for c in line if c.isalpha())
        # -> if line has less than 30% letters, it's probably garbage (ISBN, emails, phone numbers, etc)
        if len(line) > 20 and letter_count / len(line) < 0.3:
            continue
        # -> skip lines that are mostly special chars
        if len(line) > 10:
            special_chars = sum(1 for c in line if not (c.isalnum() or c.isspace() or c in 'ăâîșț.,;:!?"\'-()[]{}'))
            if special_chars / len(line) > 0.4:
                continue
        filtered_lines.append(line)
    
    text = '\n'.join(filtered_lines)
    # -> strip leading/trailing whitespace
    text = text.strip()
    return text

# -> process each file, clean the text, and return a list of records
def process_jsonl_file(file_path):
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'text' in record:
                        cleaned_text = clean_final_text(record['text'])
                        if len(cleaned_text) > 500:
                            records.append({'text': cleaned_text})
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return records

# -> process txt files (if any) and return records
def process_txt_file(file_path):
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            cleaned_text = clean_final_text(text)
            if len(cleaned_text) > 500:
                records.append({'text': cleaned_text})
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return records

# -> cleaning function to check if text is mostly garbage (too many non-ASCII chars, too many newlines, etc)
def concatenate_and_clean(file1, file2, output_file):
    all_records = []
    total_chars = 0
    
    input_files = [file1, file2]
    input_files = [f for f in input_files if os.path.exists(f)]
    print(f"found {len(input_files)} files to process")
    
    for file_path in tqdm(input_files, desc="processing files"):
        records = process_jsonl_file(file_path)
        
        for record in records:
            all_records.append(record)
            total_chars += len(record['text'])
    
    # -> write to output jsonl
    print("writing to output file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in tqdm(all_records, desc="writing records"):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # -> calculate final size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    file_size_gb = file_size_mb / 1024
    print(f"total records: {len(all_records)}")
    print(f"total characters: {total_chars:,}")
    print(f"total characters (GB): {total_chars / (1024**3):.2f}GB")
    print(f"output file size: {file_size_mb:.2f}MB ({file_size_gb:.2f}GB)")
    print(f"output file: {output_file}")
    return all_records, total_chars


if __name__ == "__main__":
    try:
        final_records = []
        total_chars = 0
        
        input_files = [CLEANED_CORPUS_FILE, OUTPUT_JSONL]
        input_files = [f for f in input_files if os.path.exists(f)]
        
        print(f"found {len(input_files)} files to concatenate")
        print()
        
        if len(input_files) == 0:
            print("error: no input files found!")
            exit(1)
        
        # -> read both files and concatenate
        for file_path in input_files:
            print(f"reading {file_path}...")
            file_records = 0
            file_chars = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in tqdm(f, desc=f"processing {os.path.basename(file_path)}"):
                        try:
                            record = json.loads(line.strip())
                            if 'text' in record and record['text']:
                                final_records.append(record)
                                file_records += 1
                                file_chars += len(record['text'])
                                total_chars += len(record['text'])
                        except (json.JSONDecodeError, ValueError):
                            continue
                
                print(f"-> records: {file_records}")
                print(f"-> characters: {file_chars:,}")            
            except Exception as e:
                print(f"error reading {file_path}: {e}")
                continue
        
        if len(final_records) == 0:
            print("error: no records found in input files!")
            exit(1)
        
        # -> write final combined file
        print("writing final combined file...")
        try:
            with open(FINAL_OUTPUT, 'w', encoding='utf-8') as f:
                for record in tqdm(final_records, desc="writing records"):
                    try:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    except Exception as e:
                        print(f"error writing record: {e}")
                        continue
        
        except Exception as e:
            print(f"error writing output file: {e}")
            exit(1)
        
        # -> final stats
        try:
            file_size_mb = os.path.getsize(FINAL_OUTPUT) / (1024 * 1024)
            file_size_gb = file_size_mb / 1024
            
            print(f"total records: {len(final_records):,}")
            print(f"total characters: {total_chars:,}")
            print(f"total characters (GB): {total_chars / (1024**3):.2f}GB")
            print(f"output file size: {file_size_mb:.2f}MB ({file_size_gb:.2f}GB)")
            print(f"output file: {FINAL_OUTPUT}")        
        except Exception as e:
            print(f"error calculating stats: {e}")
    
    except KeyboardInterrupt:
        print("operation cancel")
        exit(0)
    except Exception as e:
        print(f"fatal error: {e}")
        exit(1)