import os
import glob
import re

print("Recovering test files...")
all_files = glob.glob("tests/*.py")

# test_api.py contains the concatenated output, let's use it as source
with open("tests/test_api.py", "r", encoding="utf-8") as f:
    content = f.read()

# I will find boundaries.
# Usually a file starts with `#!/usr/bin/env python3` or `"""\ntests/` or `import ` or `def test_`
# The Get-ChildItem order is alphabetical.
expected_names = sorted([os.path.basename(f) for f in all_files])

print("Expected files:", expected_names)

# We can split by looking for `if __name__ == "__main__":\n    unittest.main()\n` and similar ends.
# But `test_abo_positive_4class.py` uses pytest so it doesn't have `unittest.main()`.

# Let's search for the start of each file.
# The concatenated file is exactly the files in alphabetical order joined together.
# If we just do this manually, we can find the exact indices.

lines = content.splitlines()

# We need to find 16 starts.
# File 0: test_abo_positive_4class.py -> starts at line 0
# Let's find all lines containing `"""\ntests/test_` or `import os` if it's right after `unittest.main()`
starts = [0]
for i in range(1, len(lines)):
    # file boundaries happen right after the previous file ends.
    # Previous file often ends with `unittest.main()` or `assert` or something.
    # But wait, looking for `import os` or `"""` that match alphabetical files is easiest.
    if lines[i].startswith('"""') and i+1 < len(lines) and lines[i+1].startswith('tests/'):
        starts.append(i)
    elif lines[i].startswith('import os') and lines[i-1] == '    unittest.main()':
        starts.append(i)
    elif lines[i].startswith('import os') and lines[i-1] == '':
        # Let's look around line 400 for test_calibration.py
        pass

# Let's just output the lines to a temp file so I can view it and write exact line numbers.
with open("tests/dump_lines.txt", "w", encoding="utf-8") as f:
    for i, l in enumerate(lines):
        f.write(f"{i}: {l}\n")

print("Dumped to tests/dump_lines.txt for manual boundary checking.")
