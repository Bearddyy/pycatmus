"""Model resolution and first-run download from Zenodo.

Models are never bundled in the wheel. They are fetched once into a local
cache (``$PYCATMUS_MODELS`` if set, else the platform user-data dir) and
verified against the md5 recorded with the Zenodo file.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from platformdirs import user_data_dir

MODEL_CACHE = Path(
    os.environ.get("PYCATMUS_MODELS", Path(user_data_dir("pycatmus")) / "models")
)

# name -> (zenodo record id, filename, md5). Pinned to specific *version*
# records (not concept DOIs) so the md5 stays valid.
_MODELS: dict[str, dict[str, str]] = {
    "catmus-medieval": {
        "record": "21488839",
        "filename": "catmus-medieval-1.6.0.mlmodel",
        "md5": "326a6ea675661ec9ce0bb6b1958de495",
        "doi": "10.5281/zenodo.21488839",
        "license": "CC-BY-4.0",
    },
    "blla": {
        "record": "22879549",
        "filename": "blla.mlmodel",
        "md5": "f3d933d00dba8b47bbc2f11804cd32f0",
        "doi": "10.5281/zenodo.14602568",
        "license": "Apache-2.0",
    },
}


def download(name: str, dest_dir: Path | None = None, quiet: bool = False) -> Path:
    """Fetch model *name* into *dest_dir*, verify md5, return its path."""
    try:
        info = _MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model '{name}' (known: {', '.join(_MODELS)})") from None
    dest_dir = dest_dir or MODEL_CACHE
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / info["filename"]
    url = f"https://zenodo.org/api/records/{info['record']}/files/{info['filename']}/content"
    if not quiet:
        print(f"downloading {info['filename']} from Zenodo {info['doi']} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL
    md5 = hashlib.md5()
    with open(tmp, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5.update(chunk)
    if md5.hexdigest() != info["md5"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"md5 mismatch downloading {info['filename']}: got {md5.hexdigest()}, "
            f"expected {info['md5']}. The download was corrupted; try again."
        )
    tmp.replace(dest)
    if not quiet:
        print(f"saved {dest}", flush=True)
    return dest


def resolve(name: str, override: Path | None = None) -> Path:
    """Return a usable path for *name*: explicit override, cache, or download.

    A local file with the model's canonical filename anywhere on the default
    kraken search paths is honoured too (``kraken get`` installs to htrmopo).
    """
    if override is not None:
        if not override.is_file():
            raise FileNotFoundError(f"model file not found: {override}")
        return override
    filename = _MODELS[name]["filename"]
    dest = MODEL_CACHE / filename
    if dest.is_file():
        return dest
    # models fetched with `kraken get` land here — reuse instead of re-downloading
    from platformdirs import user_data_dir as _udd

    for cand in (_udd("htrmopo"), Path.cwd() / filename):
        hit = Path(cand) / filename
        if hit.is_file():
            return hit
        found = sorted(Path(cand).rglob(filename)) if Path(cand).is_dir() else []
        if found:
            return found[0]
    return download(name)
