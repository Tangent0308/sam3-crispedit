"""Atomic cross-replica preferences based on accepted plans and in-flight work.

This ledger balances planning supply, not final visual quality. Natural fallback
is always allowed; rejected plans contribute no accepted count. Keep on local
disk. An interrupted owner's reservations remain visible until explicitly
cancelled, rather than silently allowing duplicate model calls.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

OPERATIONS = ('add', 'replace', 'remove', 'attribute')


class SharedOperationQuota:
    def __init__(self, path, config):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS reservations (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, owner TEXT NOT NULL,
                priority TEXT NOT NULL, preferred TEXT NOT NULL,
                state TEXT NOT NULL, actual TEXT);
            CREATE TABLE IF NOT EXISTS totals (
                operation TEXT PRIMARY KEY, accepted INTEGER NOT NULL, inflight INTEGER NOT NULL);
        ''')
        value = json.dumps(config, sort_keys=True)
        self.db.execute('INSERT OR IGNORE INTO config VALUES (1,?)', (value,))
        if self.db.execute('SELECT value FROM config WHERE id=1').fetchone()[0] != value:
            self.close()
            raise ValueError('shared quota has different planning configuration')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for op in OPERATIONS:
                self.db.execute('INSERT OR IGNORE INTO totals VALUES (?, '
                                "(SELECT COUNT(*) FROM reservations WHERE state='done' AND actual=?), "
                                "(SELECT COUNT(*) FROM reservations WHERE state='pending' AND preferred=?))",
                                (op, op, op))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def close(self):
        self.db.close()

    def counts(self):
        values = self.db.execute('SELECT * FROM totals').fetchall()
        return {field: {r['operation']: r[field] for r in values if r[field]}
                for field in ['accepted', 'inflight']}

    def reserve(self, job_id, fingerprint, owner):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            existing = self.db.execute('SELECT * FROM reservations WHERE id=?', (job_id,)).fetchone()
            if existing:
                if existing['fingerprint'] != fingerprint:
                    raise ValueError('same source/mask has different input content')
                if existing['owner'] != owner or existing['state'] != 'pending':
                    raise ValueError('source/mask already owned or completed; do not call model twice')
                priority = json.loads(existing['priority'])
            else:
                counts = self.counts()
                total = Counter(counts['accepted']) + Counter(counts['inflight'])
                priority = sorted(OPERATIONS, key=lambda op: (total[op], OPERATIONS.index(op)))
                self.db.execute('INSERT INTO reservations VALUES (?,?,?,?,?,?,NULL)',
                                (job_id, fingerprint, owner, json.dumps(priority), priority[0], 'pending'))
                self.db.execute('UPDATE totals SET inflight=inflight+1 WHERE operation=?', (priority[0],))
            self.db.execute('COMMIT')
            return priority
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def finish(self, job_id, owner, actual):
        if actual is not None and actual not in OPERATIONS:
            raise ValueError('unknown actual operation')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT * FROM reservations WHERE id=?', (job_id,)).fetchone()
            if not row or row['owner'] != owner:
                raise ValueError('reservation belongs to another owner or is missing')
            if row['state'] == 'done' and row['actual'] != actual:
                raise ValueError('completed reservation cannot change its outcome')
            if row['state'] == 'pending':
                self.db.execute('UPDATE totals SET inflight=inflight-1 WHERE operation=?', (row['preferred'],))
                if actual is not None:
                    self.db.execute('UPDATE totals SET accepted=accepted+1 WHERE operation=?', (actual,))
            self.db.execute("UPDATE reservations SET state='done',actual=? WHERE id=?", (actual, job_id))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def cancel_pending(self, owner):
        """Explicit recovery only, after confirming this owner has stopped."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for op, count in self.db.execute("SELECT preferred,COUNT(*) FROM reservations WHERE owner=? "
                                             "AND state='pending' GROUP BY preferred", (owner,)).fetchall():
                self.db.execute('UPDATE totals SET inflight=inflight-? WHERE operation=?', (count, op))
            count = self.db.execute("DELETE FROM reservations WHERE owner=? AND state='pending'", (owner,)).rowcount
            self.db.execute('COMMIT')
            return count
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
