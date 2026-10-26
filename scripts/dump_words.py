"""Print words from a transcription so a demo word can be chosen. Run as a file."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from pycatmus.core import Engine

    eng = Engine(rec_model=Path(sys.argv[1]))
    res = eng.transcribe_page(Path(sys.argv[2]))
    for i, line in enumerate(res.lines):
        words = [(w.text.strip(), w.conf) for w in line.words if w.text.strip()]
        print(i, words)


if __name__ == "__main__":
    main()
