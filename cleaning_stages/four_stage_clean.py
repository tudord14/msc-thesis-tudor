import re, json, os, argparse, hashlib
from pathlib import Path
from typing   import Iterator
from tqdm     import tqdm

def iter_text(p: Path) -> Iterator[str]:
    if p.is_dir():
        for f in p.rglob("*.jsonl"):
            yield from iter_text(f)
        return
    with p.open(encoding="utf-8") as fh:
        for ln in fh:
            try:
                obj = json.loads(ln)
                if "text" in obj:
                    yield obj["text"]
            except json.JSONDecodeError:
                continue

DATE_RE = re.compile(r'\d{2}\.\d{2}\.\d{4}')
_HAS_URL = re.compile(r'https?://\S+|www\.\S+|\b\w+\.(?:com|net|org|info|gov|edu|ro|eu|uk|de|fr|it|es|pl|cz|co)\b', re.I)
def token_count(t: str) -> int: return len(t.split())
_ADMIN_FOOTER = re.compile(r'cu\s+ducerea\s+la\s+îndeplinire.*hotărâri|administrația\s+imobiliară', re.I)
FLIGHT_RE     = re.compile(r'\bZborul\s+W6\b', re.I)
_RYANAIR_RE   = re.compile(r'\bRyanair\b.*\bZborur[i]?|\bBilete de avion\b', re.I)
_IT_BLOG_RE   = re.compile(r'\bIT\s*School\b', re.I)
_COURSE_RE    = re.compile(r'\bcurs\b.*\bincepe\b.*\bora\b', re.I)
_FORUM_REPLY  = re.compile(r'(Mesaje\s*:|Data\s+de\s+inscriere\s*:|Reputatie)', re.I)
_TRACTOR_HEAD = re.compile(r'\bTipuri\s+de\s+tractoare\b', re.I)
_TERMENE      = re.compile(r'\bTermene\.ro\b', re.I)
_SONDE_RE     = re.compile(r'\bSONDE\s*:\s*\d+', re.I)
_RUGACIUNE_RE = re.compile(r'Casa\s+de\s+rugaciune', re.I)
_ISJ_DOLJ_RE  = re.compile(r'\bISJ\s+Dolj\b.*?BAZA\s+DE\s+DATE', re.I)
_SCOALA_RE    = re.compile(r'\bSCOALA\s+CU\s+CLASELE', re.I)
_ARCHIVE_RE   = re.compile(r'\bArchives\s+-', re.I)
_HOTEL_DIST   = re.compile(r'Distanța\s+de\s+la\s+.+?km', re.I)

def admin_footer(t):   return _ADMIN_FOOTER.search(t) and token_count(t) < 40
def flight_table(t):   return (len(DATE_RE.findall(t)) >= 20 and FLIGHT_RE.search(t)) or _RYANAIR_RE.search(t)
def it_school_dump(t): return _IT_BLOG_RE.search(t) or _COURSE_RE.search(t)
def forum_reply_dump(t): return len(_FORUM_REPLY.findall(t)) >= 2
def tractor_list(t):   return _TRACTOR_HEAD.search(t) and len(re.findall(r'\d{3,4}', t)) > 100
def firm_catalog(t):   return _TERMENE.search(t) and 'LISTA FIRME' in t.upper()
def sonde_log(t):      return _SONDE_RE.search(t) and 'RELE' in t.upper()
def parish_stub(t):    return _RUGACIUNE_RE.search(t)
def isj_dolj_dropdown(t): return _ISJ_DOLJ_RE.search(t) or len(_SCOALA_RE.findall(t)) >= 40
def misc_lists(t):     return _ARCHIVE_RE.search(t) or _HOTEL_DIST.search(t)
def has_url(t):        return bool(_HAS_URL.search(t))
def brainly(t):        return 'brainly.ro' in t.lower()

def should_drop(txt: str) -> bool:
    if token_count(txt) < 75:          return True
    if brainly(txt):                   return True
    if has_url(txt):                   return True
    if admin_footer(txt):              return True
    if flight_table(txt):              return True
    if it_school_dump(txt):            return True
    if forum_reply_dump(txt):          return True
    if tractor_list(txt):              return True
    if firm_catalog(txt):              return True
    if sonde_log(txt):                 return True
    if parish_stub(txt):               return True
    if isj_dolj_dropdown(txt):         return True
    if misc_lists(txt):                return True
    return False

FLUSH_EVERY = 1_000 
def main() -> None:
    ap = argparse.ArgumentParser(description="corpus cleaner")
    ap.add_argument("input",  help="stage-3 / stage-4 source .jsonl (or folder)")
    ap.add_argument("-o", "--output", default="clean_stage4f.jsonl",
                    help="destination file (default: %(default)s)")
    args = ap.parse_args()

    src, dst = Path(args.input).expanduser(), Path(args.output).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)

    seen_sha = set()
    kept = 0

    with dst.open("w", encoding="utf-8") as fh_out:
        for raw in tqdm(iter_text(src), desc="Stage-4 f", unit="obj",
                        dynamic_ncols=True, smoothing=0.1):

            if should_drop(raw):
                continue
            sha = hashlib.sha1(raw.encode()).hexdigest()
            if sha in seen_sha:
                continue
            seen_sha.add(sha)

            json.dump({"text": raw}, fh_out, ensure_ascii=False)
            fh_out.write("\n")
            kept += 1

            if kept % FLUSH_EVERY == 0:
                fh_out.flush()
                os.fsync(fh_out.fileno())

        # final sync
        fh_out.flush()
        os.fsync(fh_out.fileno())

    print(f"\nkept {kept:,} chunks  →  {dst.resolve()}  (flushed every {FLUSH_EVERY})")

if __name__ == "__main__":
    main()
