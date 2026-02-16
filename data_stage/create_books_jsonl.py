import json
import os
from random import shuffle
import re
from pathlib import Path

books_folder = "Extracted_Texts"
books_jsonl_file = "books.jsonl"
agonia_jsonl_file = "agonia.jsonl"
final_jsonl_file = "final_high_quality.jsonl"

INPUT_DIR = "/Volumes/KINGSTON/Extracted_Texts/"
CLEANED_DIR = "/Volumes/KINGSTON/Cleaned_Texts/"
OUTPUT_JSONL = "/Volumes/KINGSTON/data.jsonl"

# -> a type of aggressive cleaning that tries to remove as much noise as possible, while keeping the text structure and formatting as much as possible
def clean_text_aggressive(text):
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = ''.join(ch for ch in text if not (ord(ch) < 32 and ch not in '\n\t\r'))
    
    # -> remove all angle brackets with garbage (OCR artifacts like <XOPT]youvtO>)
    text = re.sub(r'<[^>]*>', '', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # -> skip lines with >30% special characters
        special_count = sum(1 for c in line if not (c.isalnum() or c.isspace() or c in 'ăâîșțçéèêëàìíîïòóôõöùúûüýÿœæĂÂÎȘȚ.,;:!?"\'-()[]{}'))
        if special_count / (len(line) + 1e-9) > 0.3:
            continue
        
        if re.match(r'^[\s\*\.\-_]{5,}$', line.strip()):
            continue
        if re.match(r'^[\s\d\-–—]{0,20}$', line.strip()):
            continue
        if '....' in line or '----' in line or '***' in line:
            continue
        cleaned_lines.append(line)
    
    # -> cleaned lines are rejoined and further cleaned with regexes to remove multiple newlines, 
    # -> hyphenated words, lines that are just numbers or special characters, and excessive whitespace
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'^\s*[\d\*\-\._]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # -> skip lines with too many CJK or rare unicode
        rare_unicode = sum(1 for c in line if ord(c) > 0x3000)
        if rare_unicode / (len(line) + 1e-9) > 0.2:
            continue
        cleaned_lines.append(line)
    
    # -> rejoin cleaned lines
    text = '\n'.join(cleaned_lines)
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    # -> try to reconstruct paragraphs 
    paragraphs = []
    current_paragraph = []
    for line in lines:
        if len(line) < 15 and line.isupper():
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            paragraphs.append(line)
        else:
            current_paragraph.append(line)
    
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    text = '\n\n'.join(paragraphs)
    
    return text.strip()


# -> read, clean and write a file
def process_single_file(input_path, output_path, min_length=1000):
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    cleaned = clean_text_aggressive(text)
    if len(cleaned) < min_length:
        return False, "too_short"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    return True, len(cleaned)

# -> batch process all files in a directory, with stats and error handling
def batch_clean_files(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    txt_files = sorted(list(input_path.glob('*.txt')))
    total = len(txt_files)
    
    stats = {
        'success': 0,
        'too_short': 0,
        'error': 0,
        'total_chars': 0
    }
    
    for idx, txt_file in enumerate(txt_files):
        try:
            success, result = process_single_file(
                txt_file,
                output_path / txt_file.name
            )
            if success:
                stats['success'] += 1
                stats['total_chars'] += result
                print(f"[{idx+1}/{total}] GOOD {txt_file.name} ({result/1024:.1f}KB)")
            else:
                stats['too_short'] += 1
                print(f"[{idx+1}/{total}] BAD {txt_file.name} ({result})")
        
        except Exception as e:
            stats['error'] += 1
            print(f"[{idx+1}/{total}] BAD {txt_file.name} (error: {str(e)[:50]})")
    
    print(f"cleaned: {stats['success']} files")
    print(f"skipped (too short): {stats['too_short']} files")
    print(f"errors: {stats['error']} files")
    print(f"total size: {stats['total_chars'] / (1024**2):.1f}MB")
    return stats

# -> smart chunking that respects paragraph and sentence boundaries
def smart_chunk_text(text, chunk_size=2048):
    chunks = []
    current_chunk = ""
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        # -> if adding this paragraph would exceed chunk size, save current chunk and start new one
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph
    
    # -> add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# -> convert cleaned text files into a single jsonl file without splitting words
def convert_to_jsonl(cleaned_dir, output_jsonl, chunk_size=2048):
    cleaned_path = Path(cleaned_dir)
    txt_files = sorted(list(cleaned_path.glob('*.txt')))
    total_records = 0
    
    with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # -> use smart chunking that respects paragraph boundaries
            chunks = smart_chunk_text(text, chunk_size=chunk_size)
            
            for chunk in chunks:
                if len(chunk) > 100:
                    record = {'text': chunk}
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                    total_records += 1
    
    print(f"created {output_jsonl} with {total_records} records")


if __name__ == "__main__":
    print("clean files")
    stats = batch_clean_files(INPUT_DIR, CLEANED_DIR)
    
    print("convert to jsonl")
    convert_to_jsonl(CLEANED_DIR, OUTPUT_JSONL, chunk_size=2048)
