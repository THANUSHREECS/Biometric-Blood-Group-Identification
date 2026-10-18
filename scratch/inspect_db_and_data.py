import sqlite3
from pathlib import Path
from PIL import Image

db_path = Path('data/blood_group.db')
if db_path.exists():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print('Tables in blood_group.db:', tables)
    for t in tables:
        tname = t[0]
        cursor.execute(f"SELECT count(*) FROM [{tname}];")
        cnt = cursor.fetchone()[0]
        cursor.execute(f"PRAGMA table_info([{tname}]);")
        cols = [c[1] for c in cursor.fetchall()]
        print(f"Table {tname}: {cnt} rows, cols: {cols}")
        cursor.execute(f"SELECT * FROM [{tname}] LIMIT 3;")
        print("Sample rows:", cursor.fetchall())
    conn.close()

# Sample check on data/raw
raw_samples = list(Path('data/raw').rglob('*.BMP'))[:5]
print(f"\nTotal BMPs in data/raw: {len(list(Path('data/raw').rglob('*.BMP')))}")
for s in raw_samples:
    with Image.open(s) as im:
        print(f"Raw image {s.name}: format={im.format}, size={im.size}, mode={im.mode}")

# Sample check on data/mfs100_final
mfs_samples = list(Path('data/mfs100_final/raw').rglob('*.bmp'))[:5]
print(f"\nTotal BMPs in data/mfs100_final: {len(list(Path('data/mfs100_final/raw').rglob('*.bmp')))}")
for s in mfs_samples:
    with Image.open(s) as im:
        print(f"MFS image {s.relative_to('data/mfs100_final/raw')}: format={im.format}, size={im.size}, mode={im.mode}")
