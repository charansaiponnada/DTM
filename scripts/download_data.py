"""Download all 10 village LiDAR point cloud files from SVAMITVA portal."""
import urllib.request, os, zipfile, time, sys, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT = Path("data/input")
INPUT.mkdir(parents=True, exist_ok=True)

# State ZIP files from SVAMITVA portal (from MoPR Hackathon data PDF)
ZIP_URLS = {
    "Gujrat_Point_Cloud.zip": "https://svamitva.nic.in/DownloadPDF/TifFile/Gujrat_Point_Cloud.zip",
    "Punjab_Point_Cloud.zip": "https://svamitva.nic.in/DownloadPDF/TifFile/Punjab_Point_Cloud.zip",
    "Rajasthan_Point_Cloud.zip": "https://svamitva.nic.in/DownloadPDF/TifFile/Rajasthan_Point_Cloud.zip",
    "Tamil_Nadu_Point_Cloud.zip": "https://svamitva.nic.in/DownloadPDF/TifFile/Tamil_Nadu_Point_Cloud.zip",
    "Andaman_and_Nicobar_Islands_1.zip": "https://svamitva.nic.in/DownloadPDF/TifFile/Andaman_and_Nicobar_Islands_1.zip",
    "Andaman_and_Nicobar_Islands_2.zip": "https://svamitva.nic.in/DownloadPDF/TifFile/Andaman_and_Nicobar_Islands_2.zip",
}

# Known internal structure of each ZIP (from prior extraction)
ZIP_CONTENTS = {
    "Gujrat_Point_Cloud.zip": [
        "DEVDI_POINT CLOUD (511671).las",
        "KHAPRETA_510206.laz",
    ],
}

def sizeof_fmt(num):
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"

def download_file(url, dest_path):
    """Download a file with progress indication."""
    print(f"  Downloading {dest_path.name} ...")
    t0 = time.time()
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192 * 1024  # 8 MB chunks
            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        elapsed = time.time() - t0
                        speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
                        print(f"\r  {dest_path.name}: {sizeof_fmt(downloaded)} / {sizeof_fmt(total)} ({pct:.1f}%) @ {speed:.1f} MB/s", end="")
        elapsed = time.time() - t0
        final_size = dest_path.stat().st_size
        print(f"\r  [OK] {dest_path.name}: {sizeof_fmt(final_size)} in {elapsed:.0f}s ({final_size/elapsed/1024/1024:.1f} MB/s)")
        return True
    except Exception as e:
        print(f"\r  [FAIL] {dest_path.name}: {e}")
        return False

def extract_zip(zip_path, extract_dir):
    """Extract a ZIP file and return list of extracted LAS/LAZ files."""
    print(f"  Extracting {zip_path.name} ...")
    t0 = time.time()
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith((".las", ".laz"))]
            if not members:
                members = zf.namelist()  # extract all if no LAS found
            for member in members:
                zf.extract(member, extract_dir)
        elapsed = time.time() - t0
        extracted = [str(p) for p in extract_dir.rglob("*") if p.suffix.lower() in (".las", ".laz")]
        print(f"  [OK] Extracted {len(extracted)} files in {elapsed:.0f}s")
        for f in extracted:
            sz = os.path.getsize(f)
            print(f"       {Path(f).name} ({sizeof_fmt(sz)})")
        return extracted
    except Exception as e:
        print(f"  [FAIL] Extraction failed: {e}")
        return []

def main():
    print("=" * 60)
    print("DTM Drainage AI — Data Downloader")
    print("=" * 60)
    
    # Only download zips that don't already exist
    to_download = {}
    existing_zips = list(INPUT.glob("*.zip"))
    for fname, url in ZIP_URLS.items():
        zip_path = INPUT / fname
        if zip_path.exists():
            print(f"[SKIP] {fname} — already exists ({sizeof_fmt(zip_path.stat().st_size)})")
        else:
            to_download[fname] = url
    
    if to_download:
        print(f"\nDownloading {len(to_download)} ZIP files ...\n")
        # Download sequentially to avoid overwhelming the server
        for fname, url in to_download.items():
            success = download_file(url, INPUT / fname)
            if not success:
                print(f"  [WARN] Will continue without {fname}")
    else:
        print("All ZIPs already downloaded!")
    
    # Extract all ZIPs
    print(f"\nExtracting point cloud files ...\n")
    all_las_files = []
    for fname in ZIP_URLS:
        zip_path = INPUT / fname
        if zip_path.exists():
            extracted = extract_zip(zip_path, INPUT)
            all_las_files.extend(extracted)
    
    # Rename files with consistent convention: VILLAGE_NAME_XXXXXX.las
    print(f"\nNormalizing filenames ...")
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for f in sorted(all_las_files):
        sz = os.path.getsize(f)
        print(f"  {Path(f).name:55s} {sizeof_fmt(sz)}")
    print(f"\nTotal: {len(all_las_files)} files")
    
    # Check what's in the input directory
    print(f"\nAll files in data/input/:")
    for f in sorted(INPUT.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".las", ".laz", ".zip"):
            sz = os.path.getsize(f)
            print(f"  {f.relative_to(INPUT):60s} {sizeof_fmt(sz)}")

if __name__ == "__main__":
    main()
