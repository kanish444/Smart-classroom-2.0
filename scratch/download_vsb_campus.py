import urllib.request
import os

candidates = [
    "https://vsbec.edu.in/wp-content/uploads/2026/02/Hostel-1.jpg",
    "https://vsbec.edu.in/wp-content/uploads/2026/04/Screenshot-19.png",
    "https://vsbec.edu.in/wp-content/uploads/2026/02/image-35.png"
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for url in candidates:
    fname = os.path.basename(url)
    out_path = os.path.join("dashboard/static/images", fname)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"SUCCESS: {fname} ({len(data)} bytes)")
    except Exception as e:
        print(f"FAILED {fname}: {e}")
