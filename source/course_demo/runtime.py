"""One process-wide model and gallery, with serialized inference and content-aware keys."""
from pathlib import Path
import gc
import sys
import threading
import numpy as np
from core import make_encoder, rank, sha256


def file_identity(path, content=False):
    if not path:
        return None
    item = Path(path).resolve()
    stat = item.stat()
    return (str(item), stat.st_size, stat.st_mtime_ns, sha256(item) if content else None)


class RetrievalRuntime:
    def __init__(self, factory=make_encoder):
        self.factory = factory
        self.lock = threading.RLock()
        self.encoder = None
        self.model_key = None
        self.gallery_key = None
        self.gallery = None
        self.loads = 0
        self.index_builds = 0

    def _release(self):
        self.encoder = None
        self.gallery = None
        self.model_key = None
        self.gallery_key = None
        gc.collect()
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear(self):
        with self.lock:
            self._release()

    def retrieve(self, mode, checkpoint, repo, device, items, path, mask=None):
        with self.lock:
            if mode == 'mari' and (not checkpoint or not Path(checkpoint).is_file()):
                self._release()
                raise FileNotFoundError('MaRI weights required; no baseline fallback')
            source_key = tuple((str(p.relative_to(repo)), p.stat().st_size, p.stat().st_mtime_ns)
                               for p in sorted(Path(repo).rglob('*.py'))) if repo and mode != 'baseline' else None
            key = (mode, device if mode != 'baseline' else 'cpu',
                   file_identity(checkpoint) if mode == 'mari' else None, source_key)
            if self.model_key != key:
                self._release()
                self.encoder = self.factory(mode, checkpoint, repo, device)
                self.model_key = key
                self.loads += 1
            gallery_key = tuple((item['id'], file_identity(item['path'], content=True)) for item in items)
            if self.gallery_key != gallery_key:
                gallery = np.stack([self.encoder.encode(item['path'], 'material') for item in items])
                # Validate before publishing a new cache entry.
                rank(gallery, gallery[0])
                self.gallery = gallery
                self.gallery_key = gallery_key
                self.index_builds += 1
            order, scores = rank(self.gallery, self.encoder.encode(path, 'image', mask))
            return order, scores
