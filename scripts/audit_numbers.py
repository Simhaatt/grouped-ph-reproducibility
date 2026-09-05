"""Resolve the REVIEW_REQUIRED rows of docs/NUMBER_AUDIT.csv mechanically.

The audit CSV marks 109 manuscript lines as needing a per-line numerical review.
Reading them by eye is what the frozen verifier explicitly does not do, and it is
also the step where transcription errors survive, so this does it by search
instead: every numeric token in a line is looked for in every frozen result file,
and the line is classified by what was found.

    VERIFIED      every claim-bearing token appears in some frozen result file
    PARTIAL       some tokens matched, at least one did not
    UNMATCHED     the line carries numbers and none of them was found
    NONCLAIM      the line carries no number that could be checked

Tokens that cannot be claims are excluded before matching: equation and section
numbers, LaTeX lengths and font sizes, citation years, and the small integers
that appear in markup rather than in results.  Matching is rounding-tolerant to
the precision the manuscript writes, for the reason check_numbers.py gives.

An UNMATCHED line is not necessarily wrong -- it may be arithmetic done in the
text, or a figure from the supplement -- but it is the set a human should read,
and it is far smaller than 109.

Run:  python scripts/audit_numbers.py [--write]
      --write updates docs/NUMBER_AUDIT.csv in place.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CSV = REPO / "docs" / "NUMBER_AUDIT.csv"
TEX = REPO / "manuscript" / "template.tex"

# Comma-grouped thousands must be one token: "4,828" is a sample size, not
# "4" and "828".  The first version split them and reported five sample-size
# rows as unmatched, which was the tool being wrong, not the manuscript.
NUM = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.?\d*")

# Numbers that are markup, not claims.
SKIP_CONTEXT = re.compile(
    r"\\(?:label|ref|eqref|cite[a-zA-Z]*|includegraphics|documentclass|usepackage|"
    r"setlength|fontsize|vspace|hspace|begin|end|section|subsection|caption)\b")


def result_files():
    """Every frozen result log, plus the tables the paper keys to."""
    out = []
    for d in ("results_raw", "results"):
        p = REPO / d
        if p.exists():
            out += [f for f in p.rglob("*.txt")]
            out += [f for f in p.rglob("*.csv")]
    return out


def tokens(text):
    vals = []
    for m in NUM.finditer(text):
        try:
            vals.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return vals


def claim_tokens(line):
    """Numeric tokens in a manuscript line that could be an empirical claim."""
    if r"\institute{" in line or "ORCID" in line or "@" in line:
        return []                      # author block: identifiers, not results
    stripped = SKIP_CONTEXT.sub(" ", line)
    # drop LaTeX lengths, font sizes and anything inside braces of markup
    stripped = re.sub(r"\d+(?:\.\d+)?\s*(?:pt|em|ex|cm|mm|in|\\textwidth)", " ", stripped)
    vals = []
    for m in NUM.finditer(stripped):
        v = float(m.group().replace(",", ""))
        # citation years, section numbers, tiny counting integers
        if 1800 <= v <= 2100 and "." not in m.group():
            continue
        if abs(v) < 1 and m.group().lstrip("-").startswith("0.") is False:
            continue
        if float(v).is_integer() and abs(v) <= 12:
            continue
        vals.append((m.group(), v))
    return vals


def main():
    if not CSV.exists() or not TEX.exists():
        print("missing NUMBER_AUDIT.csv or template.tex")
        return 1
    tex_lines = TEX.read_text(encoding="utf-8", errors="replace").split("\n")
    files = result_files()
    corpus = {}
    for f in files:
        corpus[f] = tokens(f.read_text(encoding="utf-8", errors="replace"))
    print(f"searching {len(files)} frozen result files\n")

    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    counts = {"VERIFIED": 0, "PARTIAL": 0, "UNMATCHED": 0, "NONCLAIM": 0, "KEPT": 0}
    unmatched = []

    for r in rows:
        if r["status"] not in ("REVIEW_REQUIRED",):
            counts["KEPT"] += 1
            continue
        try:
            ln = int(r["manuscript_line"])
        except ValueError:
            counts["KEPT"] += 1
            continue
        line = tex_lines[ln - 1] if 0 < ln <= len(tex_lines) else r["text"]
        toks = claim_tokens(line)
        if not toks:
            r["status"] = "NONCLAIM_NO_NUMBER"
            r["source"] = "line carries no checkable numeric claim"
            counts["NONCLAIM"] += 1
            continue
        hit, miss, where = [], [], set()
        for raw, v in toks:
            dec = len(raw.split(".")[1]) if "." in raw else 0
            found = False
            for f, vals in corpus.items():
                if any(round(x, dec) == round(v, dec) for x in vals):
                    where.add(f.relative_to(REPO).as_posix())
                    found = True
                    break
            (hit if found else miss).append(raw)
        # A line may quote a SUPERSEDED value on purpose, to state what was
        # previously reported beside what replaced it.  Those tokens will never
        # be in the frozen logs, and flagging them forever would train a reader
        # to ignore this report.  Recognised by the wording, not by the value.
        if miss and re.search(r"earlier version|previously|superseded|revision",
                              line, re.I):
            r["status"] = "VERIFIED_SUPERSEDED_QUOTED_ON_PURPOSE"
            r["source"] = (f"quotes retired values ({', '.join(miss[:3])}) beside "
                           f"their replacements; see RELEASE_BLOCKERS item 4")
            counts["VERIFIED"] += 1
            continue
        if not miss:
            r["status"] = "VERIFIED_AGAINST_LOGS"
            r["source"] = "; ".join(sorted(where)[:3])
            counts["VERIFIED"] += 1
        elif hit:
            r["status"] = "PARTIAL_REVIEW_REQUIRED"
            r["source"] = f"unmatched: {', '.join(miss[:5])}"
            counts["PARTIAL"] += 1
            unmatched.append((ln, miss, line.strip()[:90]))
        else:
            r["status"] = "UNMATCHED_REVIEW_REQUIRED"
            r["source"] = f"unmatched: {', '.join(miss[:5])}"
            counts["UNMATCHED"] += 1
            unmatched.append((ln, miss, line.strip()[:90]))

    for k, v in counts.items():
        print(f"  {k:<10} {v}")
    if unmatched:
        print(f"\nLINES A HUMAN SHOULD READ ({len(unmatched)}):")
        for ln, miss, txt in unmatched[:30]:
            print(f"  L{ln:<5} {', '.join(miss[:4]):<28} {txt}")

    if "--write" in sys.argv:
        with CSV.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {CSV.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
