#!/usr/bin/env python3
"""
make_materials_from_uscode.py

Goal:
- Parse US Code XML (uscode.house.gov download) into candidate sentences, then chunks of 4-5 sentences.
- Score each chunk for:
    (a) syntactic complexity: mean dependency length (avg |tok.i - tok.head.i|)
    (b) semantic complexity: jargon usage rate (token-in-jargon / content tokens)
- quartile-split each metric into High/Low syntactic complexity , yielding 4 conditions:
    HH, HL, LH, LL
- Sample an equal number per condition and write the output to a CSV.

Example.  Run this at the command line to get 160 items from a single file

  python3 v2materials.py \
    --xml data/usc_full/usc09.xml \
    --jargon terms.txt \
    --out outputs/jan29test.csv \
    --model en_core_web_sm \
    --n_per_cell 2

    CAUTION: 40 items is a lot per title (n_per_cell), since it will result in 160 rows per title 
    (and there are 54 titles).  if you run this as is, be prepared for a lot of items !
    
Example.  run this to iterate over a set of files:
I used the below to iterate over the full uscode (minus appendices) for the experiment materials
for f in data/usc_full/*.xml; do
  echo "Running on $f"
  python3 v2materials.py \
    --xml "$f" \
    --jargon terms.txt \
    --out "outputs/candidates/$(basename "$f" .xml)_full_candidates.csv" \
    --merge_fragments \
    --chunk_min 3 \
    --chunk_max 4 \
    --chunk_stride 3 \
    --batch_size 8 \
    --max_words 250 \
    --n_per_cell 4
done

python3 v2materials.py \
    --xml data/usc_full/usc18.xml \
    --jargon terms.txt \
    --out "outputs/candidates/usc18_test_candidates.csv" \
    --merge_fragments \
    --chunk_min 3 \
    --chunk_max 4 \
    --chunk_stride 3 \
    --batch_size 8 \
    --max_words 250 \
    --n_per_cell 10


Notes:
- assumes the XML has <section> containers (true for the current US Code XML).
- if the file uses a different container tag, adjust iter_sections().
- this code includes a function to filter out amenrment notes.  if you want to keep amendment notes,
 add the following flag when running the code: --keep_amendment_notes

"""

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple, List, Optional, Dict

import pandas as pd
from lxml import etree
import spacy
from spacy.matcher import PhraseMatcher


# ----------------------------
# Light normalization
# ----------------------------
WS_RE = re.compile(r"\s+")
SECNUM_RE = re.compile(r"^\s*§\s*\d+[A-Za-z0-9\-\.]*\s*")

def localname(tag: str) -> str:
    if tag is None:
        return ""
    return tag.split("}")[-1] if "}" in tag else tag

def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = WS_RE.sub(" ", s).strip()
    s = SECNUM_RE.sub("", s).strip()
    return s

def looks_like_noise(s: str) -> bool:
    if not s:
        return True
    if len(s) < 25:
        return True
    letters = sum(ch.isalpha() for ch in s)
    if letters / max(len(s), 1) < 0.55:
        return True
    return False


# ----------------------------
# Statute-friendly sentence merging
# ----------------------------
ENUM_START_RE = re.compile(r"^\s*\(([a-zA-Z0-9ivxIVX]+)\)\s+")
CONTINUATION_START_RE = re.compile(r"^\s*(and|or)\b", re.IGNORECASE)

def is_nonfinal(sent_text: str) -> bool:
    t = sent_text.strip().lower()
    if not t:
        return False
    if t.endswith((';', ':', ',')):
        return True
    if t.endswith('; and') or t.endswith('; or'):
        return True
    if re.search(r"\b(and|or)$", t):
        return True
    return False

def starts_like_continuation(sent_text: str) -> bool:
    t = sent_text.strip()
    if not t:
        return False
    if ENUM_START_RE.match(t):
        return True
    if CONTINUATION_START_RE.match(t):
        return True
    if t[0].islower():
        return True
    return False

def merge_statutory_sents(sent_texts: List[str]) -> List[str]:
    merged: List[str] = []
    buffer = ""

    for s in sent_texts:
        s = s.strip()
        if not s:
            continue

        if not buffer:
            buffer = s
            continue

        if is_nonfinal(buffer) or starts_like_continuation(s):
            buffer = buffer + " " + s
        else:
            merged.append(buffer)
            buffer = s

    if buffer:
        merged.append(buffer)
    return merged



def make_sentence_chunks(sent_texts: List[str], min_sents: int = 4, max_sents: int = 5, stride: int = 1) -> List[Tuple[int, int, str]]:
    """Create sliding-window chunks of 3–4 sentences (can also do other lengths though)!

    Returns a list of (start_sent_idx_1based, n_sents, chunk_text).
    """
    chunks: List[Tuple[int, int, str]] = []
    n = len(sent_texts)
    if n == 0:
        return chunks

    for k in range(min_sents, max_sents + 1):
        if k <= 0:
            continue
        for i in range(0, n - k + 1, max(1, stride)):
            chunk = " ".join(s.strip() for s in sent_texts[i:i+k] if s and s.strip()).strip()
            if chunk:
                chunks.append((i + 1, k, chunk))
    return chunks

# ----------------------------
# Metrics: syntactic dependency length and jargon density

# ----------------------------
def mean_dependency_length(doc) -> float:
    dists = []
    for tok in doc:
        if tok.is_space or tok.is_punct or tok.dep_ == "punct":
            continue
        if tok.head is tok:
            continue
        dists.append(abs(tok.i - tok.head.i))
    return float(sum(dists)) / len(dists) if dists else 0.0

def jargon_rate(doc, jargon_set: set, phrase_matcher=None) -> Tuple[float, int, int, int]:
    """
    Return (rate, jargon_hits, denom_content_tokens)
    denom: alphabetic, (optionally) non-stopword, non-punct tokens
    """
    denom = sum(1 for tok in doc if tok.is_alpha and not tok.is_space and not tok.is_punct)

    hits = 0
    phrase_hits = 0
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        if not tok.is_alpha:
            continue
        #keeping stopwords in the tokenlist, but uncommment below line to remove them
        #if tok.is_stop:
         #   continue
        if tok.lemma_.lower() in jargon_set or tok.text.lower() in jargon_set:
            hits += 1

    if phrase_matcher is not None:
        matches = phrase_matcher(doc)
        # optional: prevent overlapping spans from double-counting by keeping longest, non-overlapping:
        spans = spacy.util.filter_spans([doc[start:end] for _, start, end in matches])
        phrase_hits = len(spans)

    total_hits = hits + phrase_hits
    rate = total_hits / denom if denom else 0.0
    return rate, total_hits, denom, phrase_hits

def word_count_simple(text: str) -> int:
    # Count “words” in a stable, model-independent way for length normalization.
    return len([w for w in re.split(r"\s+", text.strip()) if w])

#RESUME CODE REVIEW HERE
# ----------------------------
# XML extraction: ONE block per section
# ----------------------------
SKIP_TAGS = {
    "heading", "head", "toc", "tocItem", "note", "notes", "sourceCredit",
    "auth", "meta", "metadata", "docNumber", "title", "subtitle",
    "table", "tbody", "thead", "tr", "td", "th", "colgroup", "col",
    "footnote", "ref", "xref", "commentary",
}
PROSE_TAGS = {"p", "para", "text", "content", "chapeau"}

def extract_section_text(elem: etree._Element, tag_counter: Counter) -> str:
    parts: List[str] = []

    for node in elem.iter():
        ln = localname(node.tag)
        tag_counter[ln] += 1

        if ln in SKIP_TAGS:
            continue

        if ln in PROSE_TAGS:
            txt = " ".join(t.strip() for t in node.itertext() if t and t.strip())
            txt = normalize_text(txt)
            if txt:
                parts.append(txt)

    if not parts:
        txt = " ".join(t.strip() for t in elem.itertext() if t and t.strip())
        return normalize_text(txt)

    return "\n".join(parts)

def iter_sections(xml_path: Path, tag_counter: Counter, max_sections: Optional[int] = None) -> Iterable[Tuple[str, str]]:
    context = etree.iterparse(str(xml_path), events=("end",), recover=True, huge_tree=True)

    n_sections = 0
    for event, elem in context:
        ln = localname(elem.tag)
        tag_counter[ln] += 1

        if ln == "section":
            n_sections += 1
            if max_sections is not None and n_sections > max_sections:
                break

            sec_id = elem.get("identifier") or elem.get("id") or f"section_{n_sections}"
            sec_text = normalize_text(extract_section_text(elem, tag_counter))
            if sec_text:
                yield sec_id, sec_text

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    del context


# ----------------------------
# Conditioning + balancing
# ----------------------------
# add conditions with quartiles
def add_conditions_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    #computing quartiles on full dataset
    sem_q25 = df["jargon_rate"].quantile(0.25)
    sem_q75 = df["jargon_rate"].quantile(0.75)
    syn_q25 = df["mean_dep_len"].quantile(0.25)
    syn_q75 = df["mean_dep_len"].quantile(0.75)

    print(f"Semantic quartiles: q25={sem_q25:.6f}, q75={sem_q75:.6f}")
    print(f"Syntactic quartiles: q25={syn_q25:.6f}, q75={syn_q75:.6f}")

    # labeling things as high, low, and mid semantic and syntactic complexity
    df["sem_level"] = df["jargon_rate"].apply(
        lambda x: "low" if x <= sem_q25 else ("high" if x >= sem_q75 else "mid")
    )
    df["syn_level"] = df["mean_dep_len"].apply(
        lambda x: "low" if x <= syn_q25 else ("high" if x >= syn_q75 else "mid")
    )

    # filtering on jargon density and synactic complexity
    df = df[(df["sem_level"] != "mid") & (df["syn_level"] != "mid")].copy()

    df["condition"] = df["syn_level"].str[0].str.upper() + df["sem_level"].str[0].str.upper()


    print("Counts by condition (after double-quartile filter):")
    print(df["condition"].value_counts().sort_index())

    return df

def balanced_sample(df: pd.DataFrame, n_per_cell: int, seed: int) -> pd.DataFrame:
    groups = {k: g for k, g in df.groupby("condition")}
    needed = ["HH", "HL", "LH", "LL"]

    # ensure all present
    for k in needed:
        if k not in groups or groups[k].empty:
            raise ValueError(f"No items found for condition {k}. Try relaxing filters or using more data.")

    min_ct = min(len(groups[k]) for k in needed)
    take = min(n_per_cell, min_ct)

    out = []
    for k in needed:
        out.append(groups[k].sample(n=take, random_state=seed))
    return pd.concat(out, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

#jargon function that tolerates phrases as well as single words
def load_jargon_items(path: Path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line.lower())
    single = {x for x in items if " " not in x}
    phrases = [x for x in items if " " in x]
    return single, phrases

def main(args: argparse.Namespace) -> None:
    xml_path = Path(args.xml)
    jargon_path = Path(args.jargon)
    out_path = Path(args.out)

    if not xml_path.exists():
        raise FileNotFoundError(f"Missing XML: {xml_path}")
    if not jargon_path.exists():
        raise FileNotFoundError(f"Missing jargon list: {jargon_path}")

    #what is this doing?  a bit confused by why this works (this function returns two things)
    # also,why is this pme
    jargon_set = load_jargon_items(jargon_path)
    print(f"Loaded jargon items: {len(jargon_set):,}")

    print(f"Loading spaCy model: {args.model}")
    nlp = spacy.load(args.model)
    nlp.max_length = max(nlp.max_length, 5_000_000)

    single_jargon, phrase_jargon = load_jargon_items(jargon_path)
    print(f"Loaded jargon items: {len(single_jargon) + len(phrase_jargon)} "
      f"(single={len(single_jargon)}, phrases={len(phrase_jargon)})")

    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    if phrase_jargon:
        phrase_patterns = [nlp.make_doc(p) for p in phrase_jargon]
        phrase_matcher.add("JARGON_PHRASE", phrase_patterns)

    tag_counter = Counter()

    # 1) Extract sections
    section_rows = []
    for sec_id, sec_text in iter_sections(xml_path, tag_counter, max_sections=args.max_sections):
        section_rows.append({"section_id": sec_id, "section_text": sec_text})

    sections_df = pd.DataFrame(section_rows)
    print(f"Extracted {len(sections_df):,} sections from {xml_path.name}")

    if sections_df.empty:
        raise RuntimeError("No <section> elements were extracted. Inspect tag inventory and adjust iter_sections().")

    # 2) Parse sections -> sentences
    # on first pass: parse sections for sentence boundaries, chunk sentences, collect chunk metadata
    # on second pass: batch parse all chunks with nlp.pipe()
    sent_rows: List[Dict] = []
    all_chunks: List[Tuple[str, int, int, str, int]] = []  # (sec_id, start_idx, k, text, wc)

    docs = nlp.pipe(sections_df["section_text"].tolist(), batch_size=args.batch_size)

    for doc, row in zip(docs, sections_df.itertuples(index=False)):
        sec_id = row.section_id

        raw_sents = [s.text for s in doc.sents] if doc.has_annotation("SENT_START") else [doc.text]
        sents = merge_statutory_sents(raw_sents) if args.merge_fragments else raw_sents

        # Build sentence chunks for longer stimuli (default: 4-5 sentences).
        # For single-sentence items, run with: --chunk_min 1 --chunk_max 1
        chunks = make_sentence_chunks(
            sents,
            min_sents=args.chunk_min,
            max_sents=args.chunk_max,
            stride=args.chunk_stride,
        )

        for start_idx, k, chunk_text in chunks:
            text = normalize_text(chunk_text)
            if looks_like_noise(text):
                continue

            wc = word_count_simple(text)
            if wc < args.min_words or wc > args.max_words:
                continue

            all_chunks.append((sec_id, start_idx, k, text, wc))

    print(f"Collected {len(all_chunks):,} chunks (pre-metrics).")

    if not all_chunks:
        raise RuntimeError("No chunks survived early filtering (noise + min/max words).")

    # batch parse all chunks at once
    chunk_texts = [c[3] for c in all_chunks]
    chunk_docs = list(nlp.pipe(chunk_texts, batch_size=args.batch_size))

    # compute metrics for each chunk-doc pair
    for (sec_id, start_idx, k, text, wc), sdoc in zip(all_chunks, chunk_docs):
        mdl = mean_dependency_length(sdoc)
        jrate, jhits, jden, phrase_hits = jargon_rate(sdoc, single_jargon, phrase_matcher)

        sent_rows.append({
            "section_id": sec_id,
            "sent_idx_in_section": start_idx,
            "chunk_start_sent": start_idx,
            "chunk_n_sents": k,
            "sent_id": f"{sec_id}__s{start_idx}_k{k}",
            "sent_text": text,
            "word_count": wc,
            "mean_dep_len": mdl,
            "jargon_rate": jrate,
            "jargon_hits": jhits,
            "jargon_denom": jden,
            "phrase_hits": phrase_hits,
            "source_xml": str(xml_path.name),
        })
    df = pd.DataFrame(sent_rows)
    print(f"Kept {len(df):,} candidate sentences before length-normalization.")

    if df.empty:
        raise RuntimeError("No sentences survived filtering. Relax min/max words or noise filter.")
    
    # 3) normalizing for length
    med_wc = float(df["word_count"].median())
    lo = int(round(med_wc - args.len_window))
    hi = int(round(med_wc + args.len_window))
    df = df[(df["word_count"] >= lo) & (df["word_count"] <= hi)].copy()
    print(f"Median word_count={med_wc:.1f}. Keeping sentences with {lo}–{hi} words => {len(df):,} remain.")

    if df.empty:
        raise RuntimeError("No sentences remain after length window filter. Increase --len_window.")

    # 4a) adding condition labels
    df = add_conditions_quartiles(df)
    print("Counts by condition (pre-balance):")
    print(df["condition"].value_counts().sort_index())
    
    _AMENDMENT_PATTERNS = [
    r"\bsubstituted\b",
    r"\bamended\b",
    r"\badded\b",
    r"\bL\.\s*\w*[-–]?\d+\b",
]
    _AMENDMENT_RE = re.compile("|".join(_AMENDMENT_PATTERNS), flags=re.IGNORECASE)

    def flag_amendment_like(text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        return bool(_AMENDMENT_RE.search(text))

    def exclude_amendment_notes(df, text_col="sent_text"):
        tmp = df.copy()
        tmp["is_amendment_note"] = tmp[text_col].map(flag_amendment_like)
        excluded = tmp[tmp["is_amendment_note"]]
        kept = tmp[~tmp["is_amendment_note"]]
        return kept, excluded

    # 4b) removing amendment notes before sampling
    if not args.keep_amendment_notes:
        before = len(df)
        df, excluded = exclude_amendment_notes(df, text_col="sent_text")
        print(f"Excluded {len(excluded):,} amendment/history-like items via keyword/regex filter ({before:,} -> {len(df):,}).")
        if not excluded.empty:
            print("Top examples of excluded text:")
            for ex in excluded["sent_text"].head(5).tolist():
                print("  -", ex[:180].replace("\n", " ") + ("…" if len(ex) > 180 else ""))
        print("Counts by condition (post-amendment-filter, pre-balance):")
        print(df["condition"].value_counts().sort_index())

    # 5) taking a balanced sample
    balanced = balanced_sample(df, n_per_cell=args.n_per_cell, seed=args.seed)
    print("Counts by condition (balanced):")
    print(balanced["condition"].value_counts().sort_index())

    # 6) save output to a csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xml", required=True, help="Path to a US Code XML file (e.g., title34.xml).")
    p.add_argument("--jargon", required=True, help="Path to jargon wordlist (one item per line).")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--model", default="en_core_web_sm", help="spaCy model (start with en_core_web_sm).")

    p.add_argument("--n_per_cell", type=int, default=40, help="How many sentences per condition to sample.")
    p.add_argument("--seed", type=int, default=13)

    p.add_argument("--min_words", type=int, default=8)
    p.add_argument("--max_words", type=int, default=80)
    p.add_argument("--len_window", type=int, default=15, help="Keep sentences within +/- this many words of the median.")
    p.add_argument("--merge_fragments", action="store_true", help="Merge statute-y sentence fragments (recommended).")

    p.add_argument("--chunk_min", type=int, default=4, help="Min sentences per stimulus chunk (default 4). Use 1 for single-sentence items.")
    p.add_argument("--chunk_max", type=int, default=5, help="Max sentences per stimulus chunk (default 5).")
    p.add_argument("--chunk_stride", type=int, default=1, help="Stride for sliding-window chunking (default 1).")

    p.add_argument("--max_sections", type=int, default=None, help="For quick tests: cap number of <section> elements.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--keep_amendment_notes", action="store_true",
               help="Keep amendment/history notes (default: exclude them).")

    args = p.parse_args()
    main(args)