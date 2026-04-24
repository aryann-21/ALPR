import os

label_dir = "dataset/valid/labels"

bad_files = 0
fixed_files = 0

for file in os.listdir(label_dir):
    path = os.path.join(label_dir, file)

    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            bad_files += 1
            continue

        cls, x, y, w, h = map(float, parts)

        # check valid range
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            bad_files += 1
            continue

        new_lines.append(line)

    if len(new_lines) == 0:
        os.remove(path)
    else:
        with open(path, "w") as f:
            f.writelines(new_lines)
            fixed_files += 1

print(f"Fixed files: {fixed_files}")
print(f"Removed bad labels: {bad_files}")