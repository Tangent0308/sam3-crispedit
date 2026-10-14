"""Bounded, thread-safe decoded-source cache for independent mask planning."""
from collections import OrderedDict
from pathlib import Path
from threading import RLock

from PIL import Image


class SourceImageCache:
    def __init__(self, max_bytes=512 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.lock = RLock()

    def get(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with self.lock:
            if key in self.entries:
                self.hits += 1
                image = self.entries.pop(key)
                self.entries[key] = image
                return image.copy()
            self.misses += 1
            with Image.open(path) as source:
                image = source.convert('RGB')
            size = image.width * image.height * 3
            if size <= self.max_bytes:
                while self.entries and self.bytes + size > self.max_bytes:
                    _, old = self.entries.popitem(last=False)
                    self.bytes -= old.width * old.height * 3
                self.entries[key] = image
                self.bytes += size
            return image.copy()

    def stats(self):
        with self.lock:
            return {'hits': self.hits, 'misses': self.misses,
                    'bytes': self.bytes, 'capacity_bytes': self.max_bytes}
