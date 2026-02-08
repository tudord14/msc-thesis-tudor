import pdfplumber
import PyPDF2
from pathlib import Path
from multiprocessing import Pool, cpu_count
import re
import unicodedata
import gc
import psutil
import os
import json

# -> we have around 5000 books -> around 160GB of pdfs
# -> in this script we will try and extract text from all the books in Romanian that I have scraped off the web
# -> as some of the books are poorly scanned and formatted we will make a type of script that skips books according to some quality checks
# -> as the books are quite large we will also use multiprocessing to speed up the process

PDF_FOLDER = r"/Volumes/KINGSTON/Text_Data/libgen/"  
OUTPUT_FOLDER = r"/Volumes/KINGSTON/Extracted_Texts/"
STATS_FILE = r"/Volumes/KINGSTON/extraction_stats.txt"
CHECKPOINT_FILE = r"/Volumes/KINGSTON/extraction_checkpoint.json"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# -> skip book if short, has too many weird characters, or has too many newlines (bad formatting)
MIN_TEXT_LENGTH = 500 
MAX_NEWLINE_RATIO = 0.25  
MIN_READABLE_CHARS = 0.85 

# -> this function will load a checkpoint file that keeps track if pdfs already processed
def load_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"processed": [], "stats": {"success": 0, "skip": 0, "no_text": 0, "gibberish": 0, "error": 0}}

# -> this function will save the checkpoint after each batch is processed
def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)

# -> get a set of already extracted books to avoid reprocessing
def get_already_extracted():
    return {f.stem for f in Path(OUTPUT_FOLDER).glob("*.txt")} 

# -> check if the text of a book is mostly gibberish (bad formatting, scanned images, or corrupted)
def is_gibberish(text):
    if not text or len(text) < MIN_TEXT_LENGTH:
        return True
    
    # -> check if most characters are readable (letters, digits, common punctuation)
    problematic = sum(1 for c in text if ord(c) > 127 and not unicodedata.category(c)[0] == 'L')
    if problematic / len(text) > 0.2: 
        return True
    
    # -> check if there are too many newlines (indicates bad formatting)
    if text.count('\n') / len(text) > MAX_NEWLINE_RATIO:
        return True
    
    return False

# -> here we try and clean the text
def clean_text(text):
    if not text:
        return ""
    
    # -> remove control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t\r")
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # -> fix  split words
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # -> fix spacing
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()

def extract_with_pdfplumber(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages: 
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text if text.strip() else None
    except:
        return None

def extract_with_pypdf2(pdf_file):
    try:
        with open(pdf_file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            
            for page_num in range(len(reader.pages)): 
                try:
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + "\n"
                except:
                    continue
            
            return text if text.strip() else None
    except:
        return None


# -> main function to extract text from a single PDF and apply quality checks
def extract_single_pdf(pdf_file):
    try:
        if pdf_file.name.startswith("._"):
            return ("skip", pdf_file.name, "mac system file")
        
        file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            return ("skip", pdf_file.name, f"file too large: {file_size_mb:.1f}MB")
        
        text = extract_with_pdfplumber(pdf_file)
        if not text:
            text = extract_with_pypdf2(pdf_file)
        
        if not text:
            return ("no_text", pdf_file.name, "no extractable text")
        
        if is_gibberish(text):
            return ("gibberish", pdf_file.name, "bad formatting/corrupted")
        
        text = clean_text(text)
        output_file = Path(OUTPUT_FOLDER) / f"{pdf_file.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        return ("success", pdf_file.name, len(text))
    
    except Exception as e:
        return ("error", pdf_file.name, str(e)[:30])
    finally:
        gc.collect()

# -> main execution block to process all PDFs in parallel and summarize results
if __name__ == "__main__":
    checkpoint = load_checkpoint()
    already_extracted = get_already_extracted()
    
    pdf_files = sorted([f for f in Path(PDF_FOLDER).glob("*.pdf") if not f.name.startswith("._")])
    pdf_files_to_process = [f for f in pdf_files if f.stem not in already_extracted and f.name not in checkpoint["processed"]]
    
    total = len(pdf_files)
    already_done = len(already_extracted)
    remaining = len(pdf_files_to_process)
    
    print(f"total PDFs found: {total}")
    print(f"already extracted: {already_done}")
    print(f"remaining to process: {remaining}")
    print()
    
    if remaining == 0:
        print("all PDFs have been extracted!")
        exit()
    
    available_ram = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    print(f"available RAM: {available_ram:.1f}GB")
    
    if available_ram < 2.0:
        print("WARNING: less than 2GB RAM available, reducing batch size to 1")
        BATCH_SIZE = 1
        MAX_WORKERS = 1
    elif available_ram < 4.0:
        print("WARNING: less than 4GB RAM available, reducing batch size to 2")
        BATCH_SIZE = 2
        MAX_WORKERS = 1
    else:
        BATCH_SIZE = 4
        MAX_WORKERS = min(2, cpu_count() // 2)
    
    stats = {
        "success": [],
        "skip": [],
        "no_text": [],
        "gibberish": [],
        "error": []
    }
    
    for batch_idx in range(0, len(pdf_files_to_process), BATCH_SIZE):
        batch = pdf_files_to_process[batch_idx:batch_idx + BATCH_SIZE]
        processed_so_far = already_done + batch_idx
        print(f"processing batch {batch_idx // BATCH_SIZE + 1} ({processed_so_far}/{total} total extracted)")
        
        with Pool(MAX_WORKERS) as pool:
            results = pool.map(extract_single_pdf, batch)
        
        for status, filename, info in results:
            stats[status].append((filename, info))
            checkpoint["processed"].append(filename)
            checkpoint["stats"][status] = checkpoint["stats"].get(status, 0) + 1
        
        save_checkpoint(checkpoint)
        gc.collect()
        current_ram = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        if current_ram < 0.5:
            print(f"WARNING: RAM critically low ({current_ram:.2f}GB), pausing for 3 seconds...")
            import time
            time.sleep(3)

    print("^"*70)
    print(f"success: {len(stats['success']):5d} books extracted -> the good data")
    print(f" skip: {len(stats['skip']):5d} mac system files")
    print(f" no_text: {len(stats['no_text']):5d} scanned/unreadable")
    print(f" gibberish: {len(stats['gibberish']):5d} bad formatting")
    print(f" error: {len(stats['error']):5d} corrupted PDFs")
    print("^"*70)
    
    # -> save detailed stats to a file for manual review
    with open(STATS_FILE, "w") as f:
        f.write("extraction summary\n")
        f.write(f"total PDFs: {total}\n")
        f.write(f"successfully extracted: {len(stats['success'])}\n\n")
        
        f.write("gibberish files:\n")
        for filename, reason in stats['gibberish'][:50]:
            f.write(f" {filename}\n")
        
        f.write("\nno_text files:\n")
        for filename, reason in stats['no_text'][:50]:
            f.write(f" {filename}\n")
