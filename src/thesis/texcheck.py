"""Compile LaTeX snippets in the TDK paper's preamble (tdk/main.tex in the project folder).

Used by the tests of every exported .tex table: a table must compile with XeLaTeX and must not
be wider than the text block (no "Overfull \\hbox" in the log).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from thesis.paths import PROJECT

TDK_MAIN = PROJECT / "tdk" / "main.tex"
MIKTEX_BIN = Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64"))


def xelatex() -> str | None:
    """Path of xelatex, or None if no TeX installation is found."""
    found = shutil.which("xelatex")
    if found:
        return found
    candidate = MIKTEX_BIN / "xelatex.exe"
    return str(candidate) if candidate.exists() else None


def tdk_preamble() -> str:
    """Everything before \\begin{document} in tdk/main.tex (fonts, margins, packages)."""
    text = TDK_MAIN.read_text(encoding="utf-8")
    return text[:text.index("\\begin{document}")]


@dataclass
class Result:
    ok: bool
    overfull: list[str]
    log: str


def compile_snippet(body: str, preamble: str | None = None, timeout: int = 300) -> Result:
    """Compile `body` inside the TDK preamble with XeLaTeX; return errors and overfull boxes."""
    exe = xelatex()
    if exe is None:
        raise FileNotFoundError("xelatex not found")
    doc = (preamble if preamble is not None else tdk_preamble()) + \
        "\\begin{document}\n" + body + "\n\\end{document}\n"
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "t.tex").write_text(doc, encoding="utf-8")
        p = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", "t.tex"], cwd=tmp,
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        log_path = Path(tmp) / "t.log"
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else p.stdout
        pdf_ok = (Path(tmp) / "t.pdf").exists()
    overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)
    return Result(ok=p.returncode == 0 and pdf_ok, overfull=overfull, log=log)
