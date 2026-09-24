"""Keep current repository documentation links usable after cleanup."""
import re
from pathlib import Path
from urllib.parse import unquote


def test_repository_relative_document_links_exist():
    root = Path(__file__).resolve().parents[1]
    missing = []
    for doc in [root / 'README.md', *sorted((root / 'docs').glob('*.md'))]:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            target = unquote(target.split('#', 1)[0])
            if not target or target.startswith(('/', 'http:', 'https:')):
                continue
            if not (doc.parent / target).exists():
                missing.append((str(doc.relative_to(root)), target))
    assert not missing, missing
