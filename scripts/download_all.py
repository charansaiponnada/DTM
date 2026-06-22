"""
Sequential download + extract + cleanup for each state ZIP.
Processes ONE zip at a time to manage disk space (25 GB free).
"""
import urllib.request, zipfile, os, time, shutil
from pathlib import Path

INPUT = Path("data/input")
INPUT.mkdir(parents=True, exist_ok=True)

STATE_ZIPS = [
    ("Gujrat_Point_Cloud.zip",           "https://svamitva.nic.in/DownloadPDF/TifFile/Gujrat_Point_Cloud.zip"),
    ("Punjab_Point_Cloud.zip",           "https://svamitva.nic.in/DownloadPDF/TifFile/Punjab_Point_Cloud.zip"),
    ("Rajasthan_Point_Cloud.zip",        "https://svamitva.nic.in/DownloadPDF/TifFile/Rajasthan_Point_Cloud.zip"),
    ("Tamil_Nadu_Point_Cloud.zip",       "https://svamitva.nic.in/DownloadPDF/TifFile/Tamil_Nadu_Point_Cloud.zip"),
    ("Andaman_and_Nicobar_Islands_1.zip","https://svamitva.nic.in/DownloadPDF/TifFile/Andaman_and_Nicobar_Islands_1.zip"),
    ("Andaman_and_Nicobar_Islands_2.zip","https://svamitva.nic.in/DownloadPDF/TifFile/Andaman_and_Nicobar_Islands_2.zip"),
]

def fmt(n):
    for u in ("B","KB","MB","GB"):
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"

def download(url, dest):
    print(f"\n--- Downloading {dest.name} ---")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        got = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(8*1024*1024)
                if not chunk: break
                f.write(chunk)
                got += len(chunk)
                if total:
                    pct = got/total*100
                    spd = got/(time.time()-t0)/1024/1024
                    print(f"\r  {fmt(got)}/{fmt(total)} ({pct:.0f}%) @ {spd:.1f} MB/s", end="")
    dt = time.time()-t0
    sz = dest.stat().st_size
    print(f"\r  [OK] {fmt(sz)} in {dt:.0f}s ({sz/dt/1024/1024:.1f} MB/s)")

def extract(zip_path, extract_to):
    print(f"  Extracting {zip_path.name} ...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".las",".laz"))]
        if not members:
            members = zf.namelist()
        for m in members:
            # Extract to a flat directory (no subdirs to avoid nesting issues)
            out_path = extract_to / Path(m).name
            if out_path.exists():
                print(f"    SKIP {Path(m).name} (exists)")
                continue
            with zf.open(m) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            sz = out_path.stat().st_size
            print(f"    {Path(m).name:55s} {fmt(sz)}")
    print(f"  [OK] Extracted in {time.time()-t0:.0f}s")

def main():
    # Check existing LAS/LAZ files
    existing = {f.name for f in INPUT.iterdir() if f.suffix.lower() in (".las",".laz")}
    print(f"Existing LAS/LAZ: {existing}")
    print(f"Free space: {fmt(shutil.disk_usage(INPUT).free)}")

    for fname, url in STATE_ZIPS:
        zip_path = INPUT / fname
        
        # Check if we already have the contents
        print(f"\n{'='*60}")
        print(f"Processing: {fname}")
        print(f"{'='*60}")

        # Download if not already present
        if not zip_path.exists():
            download(url, zip_path)
        else:
            print(f"  ZIP already exists ({fmt(zip_path.stat().st_size)})")

        # Extract
        extract(zip_path, INPUT)

        # Delete ZIP to free space
        os.remove(zip_path)
        print(f"  Deleted {fname} to free space")
        
        free = shutil.disk_usage(INPUT).free
        print(f"  Free space: {fmt(free)}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL FILE LIST")
    print(f"{'='*60}")
    for f in sorted(INPUT.iterdir()):
        if f.is_file() and f.suffix.lower() in (".las",".laz"):
            print(f"  {f.name:60s} {fmt(f.stat().st_size)}")
    print(f"\nFree space: {fmt(shutil.disk_usage(INPUT).free)}")

if __name__ == "__main__":
    main()
