"""HTTP helpers with retry/backoff, hashing, manifest writing."""
from __future__ import annotations
import hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import requests

def http_get(url: str, tries: int = 5, **kw):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, timeout=kw.pop("timeout", 60), **kw)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:
            last = e
        time.sleep(2 ** i)
    raise last

def download(url: str, dest: Path, tries: int = 60, chunk: int = 1 << 20) -> Path:
    """Streamed download with resume-less retry; returns dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for i in range(tries):
        try:
            have = dest.stat().st_size if dest.exists() else 0
            hdr = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, stream=True, timeout=60, headers=hdr) as r:
                if r.status_code == 416: return dest          # already complete
                r.raise_for_status()
                if have and r.status_code != 206: have = 0     # server ignored Range: restart
                with open(dest, "ab" if have else "wb") as f:
                    for b in r.iter_content(chunk):
                        f.write(b)
            return dest
        except requests.RequestException as e:
            last = e; time.sleep(min(2 ** i, 60))
    raise last

def file_hash(path: Path, algo: str = "md5") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def update_manifest(dirpath: Path, entry: dict):
    mf = dirpath / "MANIFEST.json"
    cur = json.loads(mf.read_text()) if mf.exists() else {"files": []}
    entry = {**entry, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    cur["files"] = [e for e in cur["files"] if e.get("name") != entry.get("name")] + [entry]
    mf.write_text(json.dumps(cur, indent=2))
