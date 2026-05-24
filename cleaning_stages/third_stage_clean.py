
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


_FORUM_BTN = [
    "vizualizări","vizite pe pagina","trebuie să vă autentificați",
    "last edited by","send a private message","find more posts",
    "copyright","toate drepturile rezervate",
]
_CTA       = ["citeste mai mult","detalii","adaugă în coș","loghează-te"]
_ATTACH_RE = re.compile(r'\.(?:jpe?g|png|gif|pdf)\s*\([\d.]+\s*(?:[kmg]i?b)\)', re.I)

def strip_forum_junk(txt: str) -> str:
    out = []
    for ln in txt.splitlines():
        low = ln.lower().strip()
        if _ATTACH_RE.search(ln):           continue
        if any(w in low for w in _FORUM_BTN):continue
        if any(w in low for w in _CTA):      continue
        out.append(ln)
    return "\n".join(out).strip()


_YEAR_COL    = re.compile(r'\b20\d{2}\b')
_PRICE_RE    = re.compile(r'\b\d{1,3}(?:[.,]\d{3})*[.,]\d{2}\s*(lei|ron|eur|€|\$|dkk)\b', re.I)
_RO_DOMAIN   = re.compile(r'\b\w+\.ro\b', re.I)

_LEGAL_KW    = {"hotarare","hotărâre","ordin","ordonanța","art.","alin.","anexa","nr."}
_MOF_MARKER  = re.compile(r'\bmonitorul\s+oficial\b', re.I)
_CONTRACT    = re.compile(r'\bcontract\b', re.I)

_COMP_SUFX   = re.compile(r'\bS\.?R\.?L\.?|S\.?A\.?|SNC\b', re.I)
_DIR_KW      = re.compile(r'\blaborator|cofetarie|patiserie|magazin|restaurant\b', re.I)

_CLIMB_GRADE = re.compile(r'\b(?:[2-9]|1[0-2])[AB]?[+-]?\b|[2-7][AB]|TD|ED', re.I)
_CLIMB_KW    = re.compile(r'\b(traseu|escalad[ăa]|peretele|lc\b)', re.I)

_UTILITY     = re.compile(r'\b(rajac|distrigaz|electrica|romgaz|apa\s+nova|enel|e\.on|cez)\b', re.I)
_PAGE_TIME   = re.compile(r'page time\s*:\s*[\d.]+\s*\(s\)', re.I)

_DIC_KW      = re.compile(r'\b(definiție|paradigmă|dexonline|sinonime|antonime)\b', re.I)
_NUM_SENSE   = re.compile(r'^\d+\.\s')

_DL_WORDS    = re.compile(r'\b(download|descarc[ăa]|free|gratis|zippy|share|hotfiles|mp3|320kbps|album\.rar|file\s*share)\b', re.I)
_PARL_LINE   = re.compile(r'^\d{3,4}\.\s+\d{2}\.\d{2}\.\d{4}', re.M)

_INCI        = re.compile(r'\b(Aqua|Glycerin|Alcohol|Sodium|Parfum|Extract|Methox|Ethyl|Xanthan|Phenoxyethanol)\b', re.I)

_ANN_WORD    = re.compile(r'\banun[țt]ur[ie]\b', re.I)
_PAGE_MODIF  = re.compile(r'\bultim[ăa]\s+vez[ae]', re.I)

_SPORT_LIGA  = re.compile(r'\bLIGA\s+a\s+[IVX]+\b', re.I)
_COACERE     = re.compile(r'\bperioada\s+de\s+coacere\b', re.I)
_ANMCS       = re.compile(r'A\.?\s*N\.?\s*M\.?\s*C\.?\s*S\.?', re.I)

_PROD_TOK    = re.compile(r'\b(Cod\s+produs|SKU|Cod\s+EAN|Modelul|Contact)\b', re.I)
_SHOW_PHONE  = re.compile(r'arăt[ăa]\s+numărul\s+de\s+telefon', re.I)

_COR         = re.compile(r'\bCOR\b')
_SIXDIGIT    = re.compile(r'\b\d{6}\b')

_CONCURS     = re.compile(r'CONCURS\s+DE\s+RECRUTARE', re.I)
_FISA_POST   = re.compile(r'\bFisa\s+Post', re.I)
_JOB_DESC    = re.compile(r'Descrierea\s+jobului', re.I)
_MARRIOTT    = re.compile(r'\bMarriott\b.*job', re.I)

_JQUERY      = re.compile(r'jQuery\s*\(')
_ISOSTAR     = re.compile(r'\bIsostar\b', re.I)

_ROCHIE      = re.compile(r'rochi[ei]\s+lung[ăa]', re.I)

_TIRE_KW     = re.compile(r'\b(lățimea|latimea|înălțimea|inaltimea|diametrul)\b', re.I)
_TIRE_ROW    = re.compile(r'\b1[3456789]5\b|\b2[0-9]{2}\b')

def looks_like_official_act(t):      return sum(k in t.lower() for k in _LEGAL_KW) >= 5
def looks_like_mof_contract(t):      return _MOF_MARKER.search(t) and _CONTRACT.search(t)
def looks_like_directory(t):         return len(_COMP_SUFX.findall(t))>=3 and len(_DIR_KW.findall(t))>=4
def looks_like_climb(t):             return len(_CLIMB_GRADE.findall(t))>=15 and len(_CLIMB_KW.findall(t))>=10
def looks_like_utility(t):           return _UTILITY.search(t) and _PAGE_TIME.search(t)
def looks_like_dex(t):               return len(_DIC_KW.findall(t))>=2 and \
                                             sum(bool(_NUM_SENSE.match(l)) for l in t.splitlines())>=3
def looks_like_company_catalog(t):
    yrs = {m.group() for m in _YEAR_COL.finditer(t) if 2001 <= int(m.group()) <= 2025}
    kw  = _COMP_SUFX.search(t) and _PRICE_RE.search(t)
    return len(yrs) >= 4 or kw

def looks_like_music_dl(t):          return len(_DL_WORDS.findall(t))>=3 and "mp3" in t.lower()
def looks_like_parliament_log(t):    return sum(bool(_PARL_LINE.match(l)) for l in t.splitlines()) >= 5
def looks_like_ingredient_list(t):   return len(_INCI.findall(t)) >= 10
def looks_like_announcements(t):     return _ANN_WORD.search(t)
def looks_like_page_modif(t):        return _PAGE_MODIF.search(t)
def looks_like_sports_table(t):      return _SPORT_LIGA.search(t) and 'clasament' in t.lower()
def looks_like_nursery_desc(t):      return _COACERE.search(t) and 'soiul' in t.lower()
def looks_like_anmcs_notice(t):      return _ANMCS.search(t) and 'ordin' in t.lower()

def looks_like_product_listing(t):
    prod_tok  = _PROD_TOK.search(t)
    phone_tok = _SHOW_PHONE.search(t)
    price_tok = _PRICE_RE.search(t)
    emag_tok  = 'emag' in t.lower()
    ro_dom    = _RO_DOMAIN.search(t)
    return prod_tok or (ro_dom and (price_tok or phone_tok or emag_tok))

def looks_like_cor_table(t):         return _COR.search(t) and len(_SIXDIGIT.findall(t)) >= 10
def looks_like_job_competition(t):   return _CONCURS.search(t) or (_FISA_POST.search(t) and 'rezultat' in t.lower())
def looks_like_job_ad(t):            return _JOB_DESC.search(t) or _MARRIOTT.search(t)
def looks_like_js_dump(t):           return _JQUERY.search(t) and len(t.splitlines()) > 5
def looks_like_isostar(t):           return _ISOSTAR.search(t) and 'vitamina' in t.lower()
def looks_like_fashion_listing(t):   return _ROCHIE.search(t) and len(re.findall(r'\d{6}', t)) > 15
def looks_like_tyre_catalog(t):      return _TIRE_KW.search(t) and len(_TIRE_ROW.findall(t)) >= 8

def should_drop(txt: str) -> bool:
    if len(txt) < 50:                       return True
    if looks_like_official_act(txt):        return True
    if looks_like_mof_contract(txt):        return True
    if looks_like_directory(txt):           return True
    if looks_like_climb(txt):               return True
    if looks_like_utility(txt):             return True
    if looks_like_dex(txt):                 return True
    if looks_like_company_catalog(txt):     return True
    if looks_like_music_dl(txt):            return True
    if looks_like_parliament_log(txt):      return True
    if looks_like_ingredient_list(txt):     return True
    if looks_like_announcements(txt):       return True
    if looks_like_page_modif(txt):          return True
    if looks_like_sports_table(txt):        return True
    if looks_like_nursery_desc(txt):        return True
    if looks_like_anmcs_notice(txt):        return True
    if looks_like_product_listing(txt):     return True
    if looks_like_cor_table(txt):           return True
    if looks_like_job_competition(txt):     return True
    if looks_like_job_ad(txt):              return True
    if looks_like_js_dump(txt):             return True
    if looks_like_isostar(txt):             return True
    if looks_like_fashion_listing(txt):     return True
    if looks_like_tyre_catalog(txt):        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="corpus cleaner")
    ap.add_argument("input",  help="*.jsonl file from stage-2 or a folder")
    ap.add_argument("-o", "--output", default="clean_stage3.jsonl",
                    help="destination file (default: %(default)s)")
    ap.add_argument("--flush", type=int, metavar="N", default=10_000,
                    help="flush & fsync every N lines (default: %(default)s)")
    args = ap.parse_args()

    in_path   = Path(args.input).expanduser()
    out_path  = Path(args.output).expanduser()
    FLUSH_EVERY = max(1, args.flush)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", encoding="utf-8")

    seen_sha     = set()
    total_raw    = total_kept = written = 0

    for raw in tqdm(iter_text(in_path), desc="Stage-3", unit="obj",
                    dynamic_ncols=True, smoothing=0.1):

        total_raw += 1
        core = strip_forum_junk(raw)
        if should_drop(core):continue

        sha = hashlib.sha1(core.encode()).hexdigest()
        if sha in seen_sha:   continue
        seen_sha.add(sha)

        json.dump({"text": core}, fh, ensure_ascii=False)
        fh.write("\n")
        total_kept += 1
        written    += 1

        if written % FLUSH_EVERY == 0:
            fh.flush(); os.fsync(fh.fileno())

    fh.flush(); os.fsync(fh.fileno()); fh.close()

    print(f"\n {total_raw:,} raw -> {total_kept:,} kept   ->  {out_path.resolve()}")


if __name__ == "__main__":
    main()
