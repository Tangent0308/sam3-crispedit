"""A cached bundle must work after relocation and never overwrite another bundle."""
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path


def test_runtime_cache_relocation_reuse_and_collision(tmp_path):
    library = tmp_path / 'bundle/lib/python3.12/site-packages'
    library.mkdir(parents=True)
    (library / 'runtime_probe.py').write_text('VALUE = 123\n')
    archive = tmp_path / 'runtime.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        handle.add(tmp_path / 'bundle/lib', arcname='lib')
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = Path(str(archive) + '.sha256')
    checksum.write_text(sha + '  ' + str(archive) + '\n')
    script = Path(__file__).resolve().parents[1] / 'scripts/cache_crispedit_runtime.sh'
    cache = tmp_path / 'local_runtime'
    command = ['bash', str(script), str(archive), str(Path(sys.executable).resolve()), str(cache)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(cache / 'bin/python')
    check = subprocess.run([str(cache / 'bin/python'), '-c', 'import runtime_probe; print(runtime_probe.VALUE)'],
                           capture_output=True, text=True, check=True)
    assert check.stdout.strip() == '123'
    ready = cache / 'bundle.sha256'
    before = ready.stat().st_mtime_ns
    subprocess.run(command, capture_output=True, text=True, check=True)
    assert ready.stat().st_mtime_ns == before
    checksum.write_text('0' * 64 + '  ' + str(archive) + '\n')
    conflict = subprocess.run(command, capture_output=True, text=True)
    assert conflict.returncode != 0 and 'another bundle' in conflict.stderr
    assert ready.read_text().strip() == sha
