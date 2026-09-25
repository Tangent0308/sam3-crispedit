#!/usr/bin/env python3
"""Download and append reviewed ScaleEdit pairs (never the unfiltered dataset)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaleedit.download import main

if __name__ == '__main__':
    main()
