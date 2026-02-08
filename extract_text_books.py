import pdfplumber
import PyPDF2
from pathlib import Path
from multiprocessing import Pool, cpu_count
import re
import unicodedata

# -> we have around 5000 books -> around 160GB of pdfs
# -> in this script we will try and extract text from all the books in Romanian that I have scraped off the web
# -> as some of the books are poorly scanned and formatted we will make a type of script that skips books according to some quality checks
# -> as the books are quite large we will also use multiprocessing to speed up the process

PDF_FOLDER = r"/Volumes/KINGSTON/Text_Data/libgen/"  
OUTPUT_FOLDER = r"/Volumes/KINGSTON/Extracted_Texts/"
STATS_FILE = r"/Volumes/KINGSTON/extraction_stats.txt"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# -> skip book if short, has too many weird characters, or has too many newlines (bad formatting)
MIN_TEXT_LENGTH = 500 
MAX_NEWLINE_RATIO = 0.25  
MIN_READABLE_CHARS = 0.80 

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
    """Try pdfplumber - EXTRACT ALL PAGES"""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:  # ← REMOVED [:20] LIMIT
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text if text.strip() else None
    except:
        return None

def extract_with_pypdf2(pdf_file):
    """Fallback to PyPDF2 - EXTRACT ALL PAGES"""
    try:
        with open(pdf_file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            
            for page_num in range(len(reader.pages)):  # ← CHANGED: min(20, ...) → len(...)
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
        # -> skip some mac system files 
        if pdf_file.name.startswith("._"):
            return ("skip", pdf_file.name, "mac system file")
        
        # -> first we try to extract with pdfplumber, if it fails we fallback to pypdf2
        text = extract_with_pdfplumber(pdf_file)
        if not text:
            text = extract_with_pypdf2(pdf_file)
        
        if not text:
            return ("no_text", pdf_file.name, "no extractable text")
        
        if is_gibberish(text):
            return ("gibberish", pdf_file.name, "bad formatting/corrupted")
        
        # -> clean and save
        text = clean_text(text)
        output_file = Path(OUTPUT_FOLDER) / f"{pdf_file.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        return ("success", pdf_file.name, len(text))
    
    except Exception as e:
        return ("error", pdf_file.name, str(e)[:30])

# -> main execution block to process all PDFs in parallel and summarize results
if __name__ == "__main__":
    pdf_files = [f for f in Path(PDF_FOLDER).glob("*.pdf") if not f.name.startswith("._")]
    total = len(pdf_files)    
    print(f"found {total} PDFs")
    
    with Pool(cpu_count()) as pool:
        results = pool.map(extract_single_pdf, pdf_files)
    
    # -> results to plot at the end
    stats = {
        "success": [],
        "skip": [],
        "no_text": [],
        "gibberish": [],
        "error": []
    }
    
    for status, filename, info in results:
        stats[status].append((filename, info))

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
