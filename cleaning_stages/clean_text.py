
import re, json, sys, hashlib, unicodedata, argparse, os
from pathlib import Path
from typing import Iterator
from tqdm import tqdm

def fix_diacritics(t: str) -> str:
    return t.translate(str.maketrans({'ş': 'ș', 'Ş': 'Ș', 'ţ': 'ț', 'Ţ': 'Ț'}))

def strip_html(t: str) -> str:
    t = re.sub(r'<[^>]*>', '', t)
    return t.replace('&lt;poem&gt;', '').replace('&lt;/poem&gt;', '')

def strip_emails_urls(t: str) -> str:
    pats = [r'\b\S+@\S+\.\S+\b', r'https?://\S+', r'www\.\S+']
    for p in pats:
        t = re.sub(p, '', t)
    return t

def strip_nbsp(t: str) -> str:
    return t.replace('\u00A0', ' ').replace('&nbsp;', ' ')

def whitelist_chars(t: str) -> str:
    return re.sub(r"[^0-9A-Za-zăâîșțĂÂÎȘȚ.,;:!?()\"'–\-\n\s]", "", t)

def fix_dot_space(t: str) -> str:
    return re.sub(r'(\.)\s*([A-ZĂÂÎȘȚ])', r'\1 \2', t)

def normalize_ws(t: str) -> str:
    t = re.sub(r'\n{2,}', '<<<P>>>', t).replace('\n', ' ')
    t = re.sub(r'\s{2,}', ' ', t)
    return t.replace('<<<P>>>', '\n\n').strip()

_SPAM = {"sex", "porno", "porn", "xxx", "bbw", "milf", "lesbian", "lesbiene",
         "gay", "trannies", "bondage", "hentai", "anal", "cum", "creampie",
         "teen", "chat", "flirt", "dating", "întâlnir", "matrimonial",
         "anunțuri", "escort", "fetish", "nud", "fund", "penis", "pula",
         "hardcore"}

def rm_keyword_spam(text: str) -> str:
    def spam(ln: str) -> bool:
        w = re.findall(r'\b\w+\b', unicodedata.normalize('NFKD', ln).lower())
        if len(w) < 6:
            return False
        if sum(tok in _SPAM for tok in w) >= 4:
            return True
        titled = sum(tok[0].isupper() for tok in re.findall(r'\b\w+\b', ln))
        return titled / len(w) > 0.6 and not re.search(r'[.!?]', ln)
    return '\n'.join(l for l in text.splitlines() if not spam(l.strip()))

_AD_WORDS = {"pret", "preț", "oferte", "comparatii", "reducere", "magazine",
             "lei", "ron", "eur", "€", "în stoc", "cumpără", "comandă",
             "discount"}
_PRICE_RE = re.compile(
    r'\b\d{1,3}(?:[.,]\d{3})*[.,]\d{2}\s*(ron|lei|eur|€|\$)', re.I)

def rm_ad_lines(text: str) -> str:
    def is_ad(ln: str) -> bool:
        low = ln.lower()
        if _PRICE_RE.search(low):
            return True
        return sum(w in low for w in _AD_WORDS) >= 2
    return '\n'.join(ln for ln in text.splitlines() if not is_ad(ln.strip()))

_CODE = re.compile(r'''(?xi)
    ^\s*(function|var|const|let|class|import|export|return)\b |
    ^\s*[$#.]?\w+\s*\( | \{\s*$ | ;\s*$ | @media|@font-face|@keyframes
''')

def rm_code(text: str) -> str:
    return '\n'.join(ln for ln in text.splitlines() if not _CODE.search(ln))

def rm_table(text: str) -> str:
    def looks_tbl(ln):
        if ln.count('|') >= 2:
            return True
        if re.search(r'\[\[.*]]', ln):
            return True
        if re.fullmatch(r'[-–—]{3,}', ln):
            return True
        return False
    return '\n'.join(ln for ln in text.splitlines()
                     if not looks_tbl(ln.strip()))

def rm_noise(text: str) -> str:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if len(ln) < 5 or len(ln.split()) <= 2:
            continue
        if re.fullmatch(r'[│─┌┐└┘├┤┬┴┼═║╔╗╝╚•·◦\s]+', ln):
            continue
        if re.fullmatch(r'[0-9\s.,:/()%+\-]+', ln):
            continue
        if sum(not c.isalpha() for c in ln) / len(ln) > .8:
            continue
        alpha = re.sub(r'[^A-Za-zĂÂÎȘȚăâîșț]', '', ln)
        if alpha.isupper() and len(alpha) > 3:
            continue
        out.append(ln)
    return '\n'.join(out)

def to_chunks(text: str, max_chars: int = 3000) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+(?=[A-ZĂÂÎȘȚ])', text)
    cur, buf = [], ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 > max_chars:
            if buf:
                cur.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        cur.append(buf)
    return cur

def clean(text: str) -> str:
    text = fix_diacritics(str(text))
    text = strip_html(text)
    text = strip_emails_urls(text)
    text = strip_nbsp(text)
    text = whitelist_chars(text)
    text = fix_dot_space(text)
    text = normalize_ws(text)
    text = rm_code(text)
    text = rm_table(text)
    text = rm_keyword_spam(text)
    text = rm_ad_lines(text)
    text = rm_noise(text)
    return text

def iter_input(paths: list[Path]) -> Iterator[str]:
    for p in paths:
        if p.is_dir():
            for f in p.rglob("*.jsonl"):
                yield from iter_input([f])
        elif p.suffix == ".jsonl":
            with p.open(encoding="utf-8") as fh:
                for ln in fh:
                    try:
                        obj = json.loads(ln)
                        if "text" in obj:
                            yield obj["text"]
                    except json.JSONDecodeError:
                        continue
        else:
            yield p.read_text(encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Clean Romanian .jsonl corpora")
    ap.add_argument("paths", nargs="+",
                    help="input .jsonl file(s) or folder(s)")
    ap.add_argument("-o", "--output", default="clean_ro.jsonl",
                    help="output file")
    args = ap.parse_args()

    in_paths = [Path(p) for p in args.paths]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fh = out_path.open("w", encoding="utf-8")
    flush_every = 10_000

    seen_sha = set()
    total_raw = total_kept = written = 0

    for text in tqdm(iter_input(in_paths), desc="Cleaning", unit="obj",
                     dynamic_ncols=True, smoothing=0.1):
        total_raw += 1
        txt = clean(text)
        if len(txt) < 50:
            continue
        sha = hashlib.sha1(txt.encode()).hexdigest()
        if sha in seen_sha:
            continue
        seen_sha.add(sha)

        for chunk in to_chunks(txt):
            json.dump({"text": chunk}, fh, ensure_ascii=False)
            fh.write("\n")
            total_kept += 1
            written += 1

            if written % flush_every == 0:
                fh.flush()
                os.fsync(fh.fileno())

    fh.flush()
    os.fsync(fh.fileno())
    fh.close()

    print(f"\n{total_raw:,} raw -> {total_kept:,} cleaned parts")
    print(f"saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()
