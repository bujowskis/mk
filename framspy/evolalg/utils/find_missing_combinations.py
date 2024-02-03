import itertools
import os
import re

funcnums = ["f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f1", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]
dims = [30]
runnums = range(1, 51)  
mutsizes = [0.1]
tsizes = [2, 5]  
migration_intervals = [10]
populations = [25]
subpopsizes = [50]
pmuts = [0.8]
pxovs = [0.2]


all_combinations = set(itertools.product(funcnums, dims, runnums, mutsizes, migration_intervals, populations, subpopsizes, pmuts, pxovs, tsizes))
print(len(all_combinations))

def parse_filename(filename):
    pattern = r'cs-(f\d+)-(\d+)-(\d+)-([\d.]+)-(\d+)-(\d+)-(\d+)-([\d.]+)-([\d.]+)-(\d+).csv'
    match = re.search(pattern, filename)
    if match:
        return (match.group(1), int(match.group(2)), int(match.group(3)), float(match.group(4)), int(match.group(5)), int(match.group(6)), int(match.group(7)), float(match.group(8)), float(match.group(9)), int(match.group(10)))
    else:
        return None

directory_path = "C:/Users/sofya/Downloads/cs/"

existing_files = set()
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        # print(filename)
        params = parse_filename(filename)
        if params:
            existing_files.add(params)


print(len(existing_files))
missing_combinations = all_combinations - existing_files
sorted_missing_combinations = sorted(missing_combinations, key=lambda x: x[0])
# print(missing_combinations)
for combo in sorted_missing_combinations:
    print(combo)

