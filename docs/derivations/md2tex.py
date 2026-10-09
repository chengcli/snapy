#!/usr/bin/env python3
"""Generate the LaTeX twin of 289-covariance-x3-curved.md from the Markdown.

The twin is GENERATED, never hand-edited, so the two files cannot drift. Run
    python3 docs/derivations/md2tex.py
from the repository root after editing the Markdown.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "289-covariance-x3-curved.md"
DST = HERE / "289-covariance-x3-curved.tex"

PREAMBLE = r"""% GENERATED FROM 289-covariance-x3-curved.md BY md2tex.py -- DO NOT HAND-EDIT.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{listings}
\usepackage[hidelinks]{hyperref}
\usepackage{longtable}
\usepackage{booktabs}
\setlength{\parskip}{0.6em}
\setlength{\parindent}{0pt}
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,frame=single,
        columns=fullflexible,keepspaces=true}
\title{Issue \#289, item 2: the $O(\Delta x_1^2)$ covariance and centroid terms
in the horizontal energy flux, on curved grids and on $x_3$}
\author{snapy / chengcli --- derivation for item 2 leg (a)}
\date{}
\begin{document}
\maketitle
"""


def esc(t: str) -> str:
    """Escape LaTeX specials in prose, leaving $...$ math alone."""
    out = []
    for i, part in enumerate(t.split("$")):
        if i % 2:                       # inside inline math
            out.append("$" + part + "$")
            continue
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                     ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
            part = part.replace(a, b)
        out.append(part)
    s = "".join(out)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"`(.+?)`", r"\\texttt{\1}", s)
    return s


def convert(md: str) -> str:
    lines = md.split("\n")
    out, i = [], 0
    env = None                           # None | 'itemize' | 'table'
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):                       # verbatim block
            if env:
                out.append(r"\end{%s}" % env); env = None
            i += 1
            out.append(r"\begin{lstlisting}")
            while i < len(lines) and not lines[i].startswith("```"):
                out.append(lines[i]); i += 1
            out.append(r"\end{lstlisting}"); i += 1
            continue
        if ln.strip() == "$$":                         # display math
            if env:
                out.append(r"\end{%s}" % env); env = None
            i += 1
            body = []
            while i < len(lines) and lines[i].strip() != "$$":
                body.append(lines[i]); i += 1
            i += 1
            out.append(r"\begin{equation*}")
            out.append("\n".join(body))
            out.append(r"\end{equation*}")
            continue
        if re.match(r"^\s*\|", ln):                    # table
            rows = []
            while i < len(lines) and re.match(r"^\s*\|", lines[i]):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
            if env:
                out.append(r"\end{%s}" % env); env = None
            n = max(len(r) for r in rows)
            out.append(r"\begin{longtable}{" + "l" * n + "}")
            out.append(r"\toprule")
            for k, r in enumerate(rows):
                out.append(" & ".join(esc(c) for c in (r + [""] * (n - len(r)))) + r" \\")
                if k == 0:
                    out.append(r"\midrule")
            out.append(r"\bottomrule")
            out.append(r"\end{longtable}")
            continue
        if re.match(r"^#{1,6} ", ln):                   # heading (needs the space)
            if env:
                out.append(r"\end{%s}" % env); env = None
            lvl = len(ln) - len(ln.lstrip("#"))
            cmd = {1: "section", 2: "section", 3: "subsection",
                   4: "subsubsection"}.get(lvl, "paragraph")
            out.append("\\%s*{%s}" % (cmd, esc(ln.lstrip("# ").strip())))
            i += 1
            continue
        if ln.strip() in ("---", "***"):
            if env:
                out.append(r"\end{%s}" % env); env = None
            out.append(r"\hrulefill"); i += 1
            continue
        m = re.match(r"^\s*[-*]\s+(.*)$", ln)
        if m:
            if env != "itemize":
                if env:
                    out.append(r"\end{%s}" % env)
                out.append(r"\begin{itemize}"); env = "itemize"
            out.append(r"\item " + esc(m.group(1))); i += 1
            continue
        if not ln.strip():
            if env == "itemize":
                out.append(r"\end{itemize}"); env = None
            out.append(""); i += 1
            continue
        out.append(esc(ln)); i += 1
    if env:
        out.append(r"\end{%s}" % env)
    return "\n".join(out)


def main() -> int:
    DST.write_text(PREAMBLE + convert(SRC.read_text()) + "\n\\end{document}\n")
    print("wrote", DST.name, len(DST.read_text().splitlines()), "lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
