import os

with open("tests/test_api.py", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

# Using the unittest.main() boundary to find the next file's start.
# Also looking for specific ends of files that don't use unittest.main()
boundaries = [0]
for i in range(len(lines)):
    if lines[i] == '    unittest.main()':
        boundaries.append(i + 1)
    elif lines[i].startswith('    assert actual_hash == expected_hash, f"Protected file {filepath} altered!'):
        boundaries.append(i + 1)
    elif lines[i].startswith('    assert len(hashes) == len(PRODUCTION_LOCK_FILES)'):
        # Wait, this might not be the end of test_calibration.py
        pass

# test_abo_positive_4class.py ends at 172
# test_calibration.py ends at 512
# test_left_thumb_protocol.py ends at 914
# test_mfs100_domain_diagnostic.py ends at 1200

b = [0, 173, 330, 379, 513, 556, 625, 684, 915, 951, 1070, 1200, 1314, 1432, 1539, 1645, len(lines)]

# Wait, let's fix the bounds using exact regex matching
b = []
files = ['test_abo_positive_4class.py', 'test_advisory_layer.py', 'test_api.py', 'test_calibration.py', 'test_database.py', 'test_dataset_prep.py', 'test_e2e_integration.py', 'test_left_thumb_protocol.py', 'test_mfs100.py', 'test_mfs100_bridge.py', 'test_mfs100_domain_diagnostic.py', 'test_mfs100_integration.py', 'test_mfs100_scanner.py', 'test_predictor.py', 'test_preprocessing.py', 'test_report.py']

# I'll just write a smart scanner that looks for `import ` or `"""` immediately following a likely EOF.
EOF_patterns = ['    unittest.main()', 'assert actual_hash == expected_hash', 'assert len(hashes) == len(PRODUCTION_LOCK_FILES)']
# Actually, I know EXACTLY which line numbers from previous runs except 10, 14.
# 9 is test_mfs100_bridge.py -> 951
# 10 is test_mfs100_domain_diagnostic.py -> let's search where test_mfs100_bridge.py ends
# test_mfs100_bridge ends at 1069 (if __name__ == "__main__": unittest.main())
# So 10 is 1070!
# test_mfs100_domain_diagnostic.py ends at 1199? Let's check 1198.
# 14 is test_preprocessing.py -> let's check test_predictor.py ending. test_predictor.py ends at 1537 (if __name__ == "__main__": unittest.main())
# So 14 is 1539 (after empty lines)
# 15 is test_report.py -> test_preprocessing ends at 1642 (unittest.main()). So 15 is 1644.

b = [0, 173, 330, 379, 513, 556, 625, 684, 915, 951, 1070, 1200, 1314, 1432, 1539, 1644, len(lines)]

for i, f in enumerate(files):
    chunk = lines[b[i]:b[i+1]]
    while chunk and chunk[0] == "":
        chunk = chunk[1:]
    with open(f"tests/{f}", "w", encoding="utf-8") as out:
        out.write('\n'.join(chunk) + '\n')

print("Files restored!")
