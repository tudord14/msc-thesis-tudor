
import re, json, os, argparse, hashlib
from pathlib import Path
from typing import Iterator
from tqdm import tqdm

def iter_text(p: Path) -> Iterator[str]:
    with p.open(encoding="utf-8") as fh:
        for ln in fh:
            try:
                obj = json.loads(ln)
                if "text" in obj:
                    yield obj["text"]
            except json.JSONDecodeError:
                continue

_FORUM_BTN = [
    "vizualizări","vizite pe pagina","trebuie să vă autentificați",
    "last edited by","send a private message","find more posts",
    "copyright","toate drepturile rezervate"
]
_CTA = ["citeste mai mult","detalii","adaugă în coș","loghează-te"]
_ATTACH_RE = re.compile(r'\.(?:jpe?g|png|gif|pdf)\s*\([\d.]+\s*(?:k|m)i?b', re.I)

def strip_forum_junk(txt: str) -> str:
    keep=[]
    for ln in txt.splitlines():
        low=ln.lower().strip()
        if _ATTACH_RE.search(ln):                     continue
        if any(w in low for w in _FORUM_BTN+_CTA):    continue
        keep.append(ln)
    return "\n".join(keep).strip()

_YEAR_COL = re.compile(r'\b20\d{2}\b')

_LEGAL_KW = {"hotarare","hotărâre","ordinul","ordonanța","art.","alin.","anexa","nr."}
_COMP_SUFFIX = re.compile(r'\bS\.?R\.?L\.?|S\.?A\.?|SNC\b', re.I)
_DIR_KW = re.compile(r'\blaborator|cofetarie|patiserie|magazin|restaurant\b', re.I)
_CLIMB_GRADE = re.compile(r'\b(?:[2-9]|1[0-2])[AB]?[+-]?\b|[2-7][AB]|TD|ED', re.I)
_CLIMB_KW = re.compile(r'\b(traseu|escalad[ăa]|peretele|lc\b)', re.I)
_UTILITY = re.compile(r'\b(rajac|distrigaz|electrica|romgaz|apa\s+nova|enel|e\.on|cez)\b', re.I)
_PAGE_TIME = re.compile(r'page time\s*:\s*[\d.]+\s*\(s\)', re.I)
_DIC_KW = re.compile(r'\b(definiție|paradigmă|dexonline|sinonime|antonime)\b', re.I)
_NUM_SENSE = re.compile(r'^\d+\.\s')
_CAT_KW = re.compile(r'\b(cui|bilant|cifra de afaceri|profitabilitatea|informațiile? de contact)\b', re.I)

def looks_like_official_act(t):   return sum(k in t.lower() for k in _LEGAL_KW) >= 5
def looks_like_directory(t):      return (len(_COMP_SUFFIX.findall(t))>=3 and len(_DIR_KW.findall(t))>=4)
def looks_like_climb(t):          return (len(_CLIMB_GRADE.findall(t))>=15 and len(_CLIMB_KW.findall(t))>=10)
def looks_like_utility(t):        return bool(_UTILITY.search(t) and _PAGE_TIME.search(t))
def looks_like_dex(t):            return len(_DIC_KW.findall(t.lower()))>=2 and \
                                          sum(bool(_NUM_SENSE.match(l)) for l in t.splitlines())>=3
def looks_like_company_catalog(t):
    yrs = {m.group() for m in _YEAR_COL.finditer(t) if 2001<=int(m.group())<=2025}
    many_years = len(yrs)>=4
    kw_hit = bool(_COMP_SUFFIX.search(t) and _CAT_KW.search(t))
    return many_years or kw_hit

_DL_WORDS = re.compile(r'\b(download|descarc[ăa]|free|gratis|zippy|share|hotfiles|mp3|320kbps|album\.rar|file\s*share)\b', re.I)
def looks_like_music_dl(t: str) -> bool:
    hits = _DL_WORDS.findall(t)
    return len(hits) >= 3 and "mp3" in [h.lower() for h in hits]

def write_jsonl(it, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w",encoding="utf-8") as fh:
        for i,chunk in enumerate(it,1):
            json.dump({"text":chunk}, fh, ensure_ascii=False); fh.write("\n")
            if i%10_000==0:
                fh.flush(); os.fsync(fh.fileno())
        fh.flush(); os.fsync(fh.fileno())

def main():
    ap = argparse.ArgumentParser(description="Romanian corpus cleaner – stage 2")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="clean_stage2.jsonl")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    written = 0
    flush_every = 10_000

    with output_path.open("w", encoding="utf-8") as fh:
        for txt in tqdm(iter_text(input_path), desc="Stage-3½", unit="obj"):
            core = strip_forum_junk(txt)
            if len(core) < 50:
                continue
            if (looks_like_official_act(core) or
                looks_like_directory(core) or
                looks_like_climb(core) or
                looks_like_utility(core) or
                looks_like_dex(core) or
                looks_like_company_catalog(core) or
                looks_like_music_dl(core)):
                continue
            sha = hashlib.sha1(core.encode()).hexdigest()
            if sha in seen:
                continue
            seen.add(sha)

            # Write each cleaned object immediately
            json.dump({"text": core}, fh, ensure_ascii=False)
            fh.write("\n")
            written += 1

            # Flush to disk every N items
            if written % flush_every == 0:
                fh.flush()
                os.fsync(fh.fileno())

        # Final flush
        fh.flush()
        os.fsync(fh.fileno())

    print(f" kept {written:,} chunks → {output_path.resolve()}")

if __name__ == "__main__":
    main()
