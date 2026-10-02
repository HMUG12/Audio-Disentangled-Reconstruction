"""Extract OpenCpop segments.zip with password."""
import pyzipper
from pathlib import Path
import time

ROOT = Path(r'E:\新创意构思\新建文件夹\ADR')
ZIP = ROOT / 'segments.zip'
OUT = ROOT / 'data' / 'opencpop'
OUT.mkdir(parents=True, exist_ok=True)

print(f'Extracting {ZIP.name} ({ZIP.stat().st_size/1024**2:.1f} MB) -> {OUT}')
print(f'Password: Mmwjxhn2017')

t0 = time.time()
with pyzipper.AESZipFile(ZIP, 'r') as zf:
    zf.setpassword(b'Mmwjxhn2017')
    # Show contents first
    names = zf.namelist()
    print(f'  Files in zip: {len(names)}')
    for n in names[:5]:
        print(f'    {n}')
    if len(names) > 5:
        print(f'    ... ({len(names)-5} more)')

    zf.extractall(path=OUT)

elapsed = time.time() - t0
print(f'[OK] Extracted in {elapsed:.1f}s')
print(f'Output: {OUT}')

# 列出内容
import os
for root, dirs, files in os.walk(OUT):
    depth = root.replace(str(OUT), '').count(os.sep)
    indent = ' ' * 2 * depth
    print(f'{indent}{os.path.basename(root)}/')
    for f in sorted(files)[:5]:
        size = (Path(root) / f).stat().st_size
        print(f'{indent}  {f} ({size:,} bytes)')
    if len(files) > 5:
        print(f'{indent}  ... ({len(files)-5} more files)')
    if depth > 2:
        break
