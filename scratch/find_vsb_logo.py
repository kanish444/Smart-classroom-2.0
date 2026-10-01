import re

path = r"C:\Users\kanis\.gemini\antigravity-ide\brain\4da4a3b2-4cb6-4c67-b1c4-2a62930a145b\.system_generated\steps\314\content.md"
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

urls = re.findall(r"https?://[^\s\"\'\<\>\)]+\.(?:png|jpg|jpeg|svg|webp)", text)
print(f"Total image URLs found: {len(urls)}")
matches = []
for u in set(urls):
    if any(k in u.lower() for k in ["logo", "vsb", "header", "brand", "crest", "icon"]):
        matches.append(u)

for m in sorted(matches):
    print(m)
