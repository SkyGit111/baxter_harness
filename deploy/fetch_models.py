"""Stage both pinned model snapshots. Can run on a workstation with Python 3 and curl."""
import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def matches(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    lfs = len(item["hash"]) == 64
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(("blob %d\0" % item["size"]).encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == item["hash"]


def fetch_file(snapshot, folder, verify_only, item):
    name = item["name"]
    if Path(name).name != name:
        raise ValueError("Expected flat model snapshot")
    target = folder / name
    if matches(target, item):
        return
    if verify_only:
        raise ValueError("Missing or damaged file: " + str(target))
    part = folder / (name + ".part")
    if not matches(part, item):
        if part.exists() and part.stat().st_size >= item["size"]:
            part.unlink()
        url = "https://huggingface.co/{}/resolve/{}/{}".format(snapshot["repo"], snapshot["revision"], name)
        print("Fetching", name, flush=True)
        for attempt in range(4):
            result = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                                     "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "90",
                                     "--continue-at", "-", "--output", str(part), url])
            if result.returncode == 0 or (part.exists() and part.stat().st_size == item["size"]):
                break
            if result.returncode == 33 and part.exists():
                part.unlink()  # This server refused a Range request. Restart just this file.
            if attempt == 3:
                result.check_returncode()
            time.sleep(2 ** attempt)
        if not matches(part, item):
            raise ValueError("Incomplete or damaged model file: " + name)
    part.replace(target)
    print("Verified", target, flush=True)


if __name__ == "__main__":
    snapshots = json.loads(Path(__file__).with_name("models.lock.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="models", type=Path)
    parser.add_argument("--only", choices=list(snapshots))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for name, snapshot in snapshots.items():
        if args.only == name or (args.only is None and not snapshot.get("optional")):
            folder = args.dir / name
            folder.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [pool.submit(fetch_file, snapshot, folder, args.verify_only, item) for item in snapshot["files"]]
                for job in jobs:
                    job.result()
            print(name, "snapshot verified", flush=True)
