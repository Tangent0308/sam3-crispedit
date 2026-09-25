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


def test_four_node_entry_is_safe_to_retry_after_clone():
    root = Path(__file__).resolve().parents[1]
    bootstrap = (root / 'scripts' / 'bootstrap_crispedit_4node.sh').read_text()
    guide = (root / 'docs' / 'CRISPEDIT_4NODE.md').read_text()

    # Bootstrap operates inside the caller's clone; it must not run the legacy
    # shared-repository setup that rejected the clone created by the entrypoint.
    assert 'Repository exists; use a new run' not in bootstrap
    assert 'repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)' in bootstrap

    # The documented worker command accepts an intact retry clone, but verifies
    # it before updating to an exact detached commit.
    assert '[[ -d $CRISPEDIT_REPO_DIR/.git ]]' in guide
    assert 'git remote get-url origin' in guide
    assert 'git diff --quiet && git diff --cached --quiet' in guide
    assert 'git fetch --no-tags origin "$CRISPEDIT_BRANCH"' in guide
    assert 'git checkout --detach "$target_commit"' in guide
