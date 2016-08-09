"""Quick script to note total runtime with notes."""

import h5py
import re

filepath = r'C:\Users\a-amb\OneDrive - University Of Cambridge\data\useful ' \
           r'prof_tests\8-8-16\experiments.py-2016_08_08_11_05_05.prof'
notes = ''
f = open(filepath)
runtime_str = re.findall(re.compile(r'Total time\: \d+\.\d+ s'), f.read())
if len(runtime_str) == 1:
    tot_runtime = runtime_str[0][11:].replace('s', '').strip()
    filename = filepath.split('/')[-1]
    writefile = h5py.File('runtimes.h5py')






