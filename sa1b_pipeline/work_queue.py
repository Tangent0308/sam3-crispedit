"""Local SQLite queue: transactional claims, resumable results, stale-worker fencing.

Place the database on local storage, not an object-store/network mount. Payloads
may reference images on shared storage. Different experiments use different DBs.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path


class WorkQueue:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending', lease TEXT, expires REAL,
                attempts INTEGER NOT NULL DEFAULT 0, result TEXT, created_at REAL);
            CREATE INDEX IF NOT EXISTS jobs_state_idx ON jobs(state);
        ''')
        # Existing completed experiments remain readable after the migration.
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if 'created_at' not in {r[1] for r in self.db.execute('PRAGMA table_info(jobs)')}:
                self.db.execute('ALTER TABLE jobs ADD COLUMN created_at REAL')
            self.db.execute('UPDATE jobs SET created_at=? WHERE created_at IS NULL', (time.time(),))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def close(self):
        self.db.close()

    def bind_config(self, config):
        value = json.dumps(config, sort_keys=True)
        self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('config', value))
        existing = self.db.execute('SELECT value FROM metadata WHERE key=?', ('config',)).fetchone()[0]
        if existing != value:
            raise ValueError('queue already bound to different model/prompt/runtime configuration')

    def enqueue(self, job):
        payload = json.dumps(job, ensure_ascii=False, sort_keys=True)
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            existing = self.db.execute('SELECT fingerprint FROM jobs WHERE id=?', (job['job_id'],)).fetchone()
            if existing:
                if existing[0] != fingerprint:
                    raise ValueError('job ID reused with different content: ' + job['job_id'])
                inserted = False
            else:
                if self.sealed():
                    raise ValueError('queue is sealed')
                self.db.execute('INSERT INTO jobs(id,fingerprint,payload,created_at) VALUES (?,?,?,?)',
                                (job['job_id'], fingerprint, payload, time.time()))
                inserted = True
            self.db.execute('COMMIT')
            return inserted
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def seal(self):
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('sealed','true')")

    def sealed(self):
        return self.db.execute("SELECT 1 FROM metadata WHERE key='sealed'").fetchone() is not None

    def claim(self, count, *, lease_seconds=3600, min_count=1, max_wait_seconds=0):
        if count < 1 or lease_seconds <= 0:
            raise ValueError('positive claim size and lease required')
        if not 1 <= min_count <= count or max_wait_seconds < 0:
            raise ValueError('invalid batch wait policy')
        now = time.time()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            # Two bounded state-index scans avoid walking all completed jobs
            # (or sorting all pending jobs) for every claim in a 50k-row queue.
            fields = 'SELECT rowid AS queue_order,id,payload,created_at FROM jobs '
            pending = self.db.execute(fields+"WHERE state='pending' ORDER BY rowid LIMIT ?", (count,)).fetchall()
            expired = self.db.execute(fields+"WHERE state='running' AND expires < ? ORDER BY rowid LIMIT ?",
                                      (now, count)).fetchall()
            rows = sorted([*pending, *expired], key=lambda r: r['queue_order'])[:count]
            # Do not reserve partial batches while waiting: multiple consumers
            # must not each hoard a fraction of the next full batch.
            if (rows and len(rows) < min_count and not self.sealed()
                    and now-min(r['created_at'] for r in rows) < max_wait_seconds):
                self.db.execute('COMMIT')
                return []
            claimed = []
            for row in rows:
                lease = uuid.uuid4().hex
                self.db.execute("UPDATE jobs SET state='running',lease=?,expires=?,attempts=attempts+1 WHERE id=?",
                                (lease, now+lease_seconds, row['id']))
                claimed.append((json.loads(row['payload']), lease))
            self.db.execute('COMMIT')
            return claimed
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def finish(self, job_id, lease, result):
        if result['job_id'] != job_id:
            raise ValueError('result belongs to another job')
        cursor = self.db.execute("UPDATE jobs SET state='done',result=?,lease=NULL,expires=NULL "
                                 "WHERE id=? AND state='running' AND lease=?",
                                 (json.dumps(result, ensure_ascii=False), job_id, lease))
        if cursor.rowcount != 1:
            raise RuntimeError('stale or duplicate worker cannot commit result')

    def release(self, job_id, lease):
        self.db.execute("UPDATE jobs SET state='pending',lease=NULL,expires=NULL "
                        "WHERE id=? AND state='running' AND lease=?", (job_id, lease))

    def counts(self):
        return dict(self.db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state').fetchall())

    def results(self):
        return list(self.iter_results())

    def completed_for(self, job_ids):
        """Read only a supervisor's bounded outstanding set, including out-of-order finishes."""
        ids = list(job_ids)
        results = []
        for start in range(0, len(ids), 500):
            batch = ids[start:start+500]
            marks = ','.join('?' for _ in batch)
            results.extend(json.loads(row[0]) for row in self.db.execute(
                f"SELECT result FROM jobs WHERE state='done' AND id IN ({marks})", batch))
        return results

    def iter_results(self):
        for row in self.db.execute("SELECT result FROM jobs WHERE state='done' ORDER BY rowid"):
            yield json.loads(row[0])

    def export_results(self, path):
        """Stream a complete queue to an atomic manifest without a 50k-row list."""
        if not self.sealed() or any(self.counts().get(state, 0) for state in ('pending', 'running')):
            raise ValueError('cannot export an unsealed or incomplete queue')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name+'.tmp')
        with temp.open('w') as stream:
            for result in self.iter_results():
                stream.write(json.dumps(result, ensure_ascii=False)+'\n')
        temp.replace(path)
