import glob
import re
import os

limitation_text = 'LIMITATION: System evaluated on 101 participants (303 scans). Accuracy is 27.72%, compared to 36.63% majority baseline. Cannot separate absence of a biological signal from scanner/domain shift.'

# Update HTML footers
for path in glob.glob('templates/*.html'):
    with open(path, 'r', encoding='utf-8') as f:
        c = f.read()
    if '<footer' in c:
        c = re.sub(r'(<footer class="app-footer">\s*<p>.*?)(</p>\s*</footer>)', r'\1 | ' + limitation_text + r'\2', c, flags=re.DOTALL)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(c)

# Fix overclaims in markdown
md_files = glob.glob('**/*.md', recursive=True)
for p in md_files:
    if os.path.isfile(p):
        with open(p, 'r', errors='ignore') as f:
            c = f.read()
        c = re.sub(r'(?i)\bproves\b', 'suggests', c)
        c = re.sub(r'(?i)\bproven\b', 'supported', c)
        c = re.sub(r'(?i)\bdefinitive\b', 'exploratory', c)
        c = re.sub(r'(?i)\bunequivocally\b', 'strongly', c)
        c = re.sub(r'(?i)\bvalidates\b', 'evaluates', c)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(c)

print('Updated docs and footers')
