#!/usr/bin/env python3
"""Run one rank of the shared-filesystem ScaleEdit labeling pipeline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaleedit.distributed import main

if __name__ == '__main__':
    main()
