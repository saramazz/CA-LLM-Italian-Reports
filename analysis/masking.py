"""First implementation of the masking patterns, used for note selection only; masked text
analysed in the paper is produced with leakage_spec.mask_text.
"""
import re
from dataclasses import dataclass, field

PLACEHOLDER = "[TERMINE RIMOSSO]"

# --- Family 1: explicit diagnostic terms -----------------------------------
# Bare "AL" collides with the Italian preposition "al" (= "to the") and the
# article; per the pre-registered plan it is never auto-masked. Instead,
# occurrences of a standalone "AL" that are NOT already part of another
# explicit-term match are logged as adjudication candidates (see
# `find_al_candidates`) for manual review.
EXPLICIT_PATTERNS = [
    r"amiloidos\w*",
    r"amiloide\w*",
    r"amyloid\w*",
    r"\bATTRwt\b",
    r"\bATTRv\b",
    r"\bATTR\b",
    r"\bTTR\b",
    r"transtiretina\w*",
    r"caten[ae]\s+legger[ei]\s+liber[ei]",
    r"\bFLC\b",
]
EXPLICIT_RE = re.compile("|".join(f"(?:{p})" for p in EXPLICIT_PATTERNS), flags=re.IGNORECASE)

# --- Family 2: disease-specific investigations and treatments ---------------
INVESTIGATION_PATTERNS = [
    r"tafamidis",
    r"patisiran",
    r"vutrisiran",
    r"inotersen",
    r"eplontersen",
    r"diflunisal",
    r"daratumumab",
    r"acoramidis",
    r"\bperugini\b",
    r"\bDPD\b",
    r"\bPYP\b",
    r"\bHMDP\b",
    r"scintigrafia\s+ossea",
    r"rosso\s+congo",
    r"immunofissazione",
    r"biopsia\s+endomiocardica",
    r"grasso\s+periombelicale",
]
INVESTIGATION_RE = re.compile("|".join(f"(?:{p})" for p in INVESTIGATION_PATTERNS), flags=re.IGNORECASE)

# --- Family 3: negated diagnostic references --------------------------------
# These leak in the opposite direction (allow inferring absence of disease).
# Bounded `.{0,N}` gaps keep matches from swallowing unrelated text between
# the negation cue and the diagnostic term.
NEGATED_PATTERNS = [
    r"amiloidosi\s+esclus[ae]",
    r"esclus[ae]\s+amiloidosi",
    r"\bnon\b.{0,30}?amiloidos\w*",
    r"scintigrafia.{0,40}?negativ[ae]",
    r"perugini\s+grado\s*0\b",
    r"assenza\s+di\s+amiloidosi",
]
NEGATED_RE = re.compile("|".join(f"(?:{p})" for p in NEGATED_PATTERNS), flags=re.IGNORECASE)

FAMILY_PATTERNS = {
    "explicit": EXPLICIT_RE,
    "investigation": INVESTIGATION_RE,
    "negated": NEGATED_RE,
}

# Standalone "AL" not already covered by an explicit-term match (e.g. "AL amiloidosi"
# is caught by EXPLICIT_RE's "amiloidos\w*" already, so it is not a candidate).
AL_CANDIDATE_RE = re.compile(r"\bAL\b")


@dataclass
class MaskResult:
    masked_text: str
    counts_by_family: dict = field(default_factory=dict)  # family -> match count
    al_candidate_count: int = 0  # standalone "AL" not already masked, for adjudication


def _merge_spans(spans):
    """Merge overlapping/adjacent (start, end) spans so a placeholder is not
    inserted twice for text matched by more than one family."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def count_matches(text: str) -> dict:
    """Per-family match counts on the given text, independent of each other
    (a span matched by two families counts once in each family's tally)."""
    if not isinstance(text, str) or not text:
        return {family: 0 for family in FAMILY_PATTERNS}
    return {family: len(pattern.findall(text)) for family, pattern in FAMILY_PATTERNS.items()}


def find_al_candidates(text: str) -> int:
    """Count standalone 'AL' occurrences not already inside an explicit-term
    match span, for manual adjudication (never auto-masked)."""
    if not isinstance(text, str) or not text:
        return 0
    explicit_spans = [m.span() for m in EXPLICIT_RE.finditer(text)]
    n = 0
    for m in AL_CANDIDATE_RE.finditer(text):
        s, e = m.span()
        if not any(es <= s and e <= ee for es, ee in explicit_spans):
            n += 1
    return n


def mask_text(text: str) -> MaskResult:
    """Mask all three term families in `text`, replacing matched spans with
    PLACEHOLDER. Returns the masked text plus per-family match counts and the
    count of standalone 'AL' adjudication candidates (never auto-masked)."""
    if not isinstance(text, str) or not text:
        return MaskResult(masked_text=text, counts_by_family={f: 0 for f in FAMILY_PATTERNS}, al_candidate_count=0)

    counts = count_matches(text)
    al_candidates = find_al_candidates(text)

    spans = []
    for pattern in FAMILY_PATTERNS.values():
        spans.extend(m.span() for m in pattern.finditer(text))
    spans = _merge_spans(spans)

    if not spans:
        return MaskResult(masked_text=text, counts_by_family=counts, al_candidate_count=al_candidates)

    out = []
    prev_end = 0
    for start, end in spans:
        out.append(text[prev_end:start])
        out.append(PLACEHOLDER)
        prev_end = end
    out.append(text[prev_end:])
    masked_text = "".join(out)

    return MaskResult(masked_text=masked_text, counts_by_family=counts, al_candidate_count=al_candidates)
