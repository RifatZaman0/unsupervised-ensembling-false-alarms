"""Download raw data and check it against the committed checksums.

    python -m thesis.download            # download what is missing, then verify
    python -m thesis.download --verify   # verify only
    python -m thesis.download --write-manifest   # record checksums (only when adding a source)

ADBench is fetched from a pinned commit of github.com/Minqi824/ADBench, ULB from Kaggle
(mlg-ulb/creditcardfraud, needs a Kaggle API token).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request

from thesis.paths import DATA_RAW, REPO

ADBENCH_COMMIT = "3dac8221081e190f157d78e93bfa8867f90d0965"
ADBENCH_URL = f"https://raw.githubusercontent.com/Minqi824/ADBench/{ADBENCH_COMMIT}/adbench/datasets/Classical/"
MANIFEST = REPO / "data" / "manifest.json"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[str, str]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["sha256"]


def download(manifest: dict[str, str]) -> None:
    for rel in manifest:
        path = DATA_RAW / rel
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.startswith("adbench/"):
            print("downloading", rel)
            urllib.request.urlretrieve(ADBENCH_URL + path.name, path)
        elif rel == "ulb/creditcard.csv":
            print("downloading ULB from Kaggle")
            subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d",
                            "mlg-ulb/creditcardfraud", "-p", str(path.parent), "--unzip"], check=True)


def verify(manifest: dict[str, str]) -> list[str]:
    """Return the files that are missing or whose checksum differs."""
    return [rel for rel, digest in manifest.items()
            if not (DATA_RAW / rel).exists() or sha256(DATA_RAW / rel) != digest]


def write_manifest() -> None:
    files = sorted(DATA_RAW.glob("adbench/*.npz")) + [DATA_RAW / "ulb" / "creditcard.csv"]
    MANIFEST.write_text(json.dumps({
        "adbench_commit": ADBENCH_COMMIT,
        "ulb_source": "kaggle:mlg-ulb/creditcardfraud",
        "sha256": {p.relative_to(DATA_RAW).as_posix(): sha256(p) for p in files},
    }, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    manifest = load_manifest()
    if not args.verify:
        download(manifest)
    bad = verify(manifest)
    print("all files match manifest" if not bad else f"missing or changed: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
