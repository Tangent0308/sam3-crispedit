#!/usr/bin/env python3
"""Complete four-node CrispEdit labeling entrypoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crispedit.distributed import main

if __name__ == '__main__':
    main()
