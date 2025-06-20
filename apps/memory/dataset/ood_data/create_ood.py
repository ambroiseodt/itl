# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Generate a list of first names that are not contained in the first_names.txt file.

### Notes
For simplicity, we force that the created file has same size than first_names.txt.

@ 2025, Meta
"""

import os
import random

iid_names = open("apps/memory/dataset/atoms/first_names.txt")
long_names = open("apps/memory/dataset/atoms/first_names_extended.txt")
iid_lines = iid_names.readlines()
iid_size = len(iid_lines)
long_lines = long_names.readlines()

# Random shuffle to break the alphabetical order
seed = 42
random.seed(seed)
random.shuffle(long_lines)

# Close files
iid_names.close()
long_names.close()

# Initialize new file
ood_file = "apps/memory/dataset/atoms/first_names_ood.txt"

# Delete ood file if it exists
if os.path.isfile(ood_file):
    os.remove(ood_file)

# Fill ood_file with names not present in iid_names
comp = 0
for line in long_lines:
    # Add names if not contained
    if line not in iid_lines:
        # Add names while ood_file is smaller than iid_names
        if comp < iid_size:
            with open(ood_file, "a") as f:
                f.write(line)
                comp += 1

# Put alphabetical order back
ood_names = open(ood_file)
lines = ood_names.readlines()
lines.sort()
os.remove(ood_file)
for line in lines:
    with open(ood_file, "a") as f:
        f.write(line)
