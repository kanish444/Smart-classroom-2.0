import urllib.request
import os

os.makedirs("dashboard/static/images", exist_ok=True)

test_urls = [
    "https://vsbec.edu.in/wp-content/uploads/2026/06/cropped-VSB-Karur-1-270x270.png",
    "https://vsbec.edu.in/wp-content/uploads/2026/06/cropped-VSB-Karur-1-192x192.png",
    "https://vsbec.edu.in/wp-content/uploads/2026/02/Layer_1-1.png",
    "https://vsbec.edu.in/wp-content/uploads/2026/02/Layer_1-2.png",
    "https://vsbec.edu.in/wp-content/uploads/2026/02/Layer_1-3.png",
    "https://vsbec.edu.in/wp-content/uploads/revslider/slider-1/VSB-banner-1-1.png",
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for url in test_urls:
    fname = os.path.basename(url)
    out_path = os.path.join("dashboard/static/images", fname)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"SUCCESS: {fname} ({len(data)} bytes)")
    except Exception as e:
        print(f"FAILED {fname}: {e}")
