"""Write the paper's LaTeX tables and PDF figures from one saved experiment run.

Reads saved artifacts only: no LLM calls, no training. Works on pilots and full/.
Usage: python scripts/paper_results.py artifacts/experiments/<protocol>/<run>
"""

import re

from run_baseline import ROOT

OUT = ROOT / "paper/generated"
SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex(value):
    return "".join(SPECIAL.get(c, c) for c in str(value))


def plain(markdown):
    """Drop Markdown emphasis, headings and bullets so responses read as prose."""
    text = re.sub(r"[*#`]+", "", markdown)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    return " ".join(text.split())


def table(name, caption, header, rows, align=None):
    """Header and rows are LaTeX already; callers tex() free text."""
    align = align or "l" + "r" * (len(header) - 1)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}\RunTag}}",
        rf"\label{{tab:{name}}}",
        r"\end{table}",
    ]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def pending(name, caption, reason):
    table(
        name,
        caption,
        [r"\textbf{Status}"],
        [[r"\textit{Pending: " + tex(reason) + "}"]],
        "l",
    )


def neighbour_agreement(vectors, keys, k=10):
    """Mean share of each item's k nearest cosine neighbours sharing at least one key."""
    import numpy as np

    x = np.asarray(vectors, dtype=float)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    k = min(k, len(x) - 1)
    nearest = np.argsort(-similarity, axis=1)[:, :k]
    return float(
        np.mean(
            [
                sum(bool(keys[i] & keys[j]) for j in row) / k
                for i, row in enumerate(nearest)
            ]
        )
    )


def movie_keys(row):
    def clean(values):
        return {v for v in values if v and v.casefold() != "unknown"}

    genres = (row["genres"] or row["Genres"] or "").split("|")
    return {
        "director": clean([row["director"]]),
        "actor": clean((row["actors"] or "").split("|")[:3]),
        "genre set": clean(["|".join(sorted(g for g in genres if g))]),
    }
