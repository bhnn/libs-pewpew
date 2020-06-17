from collections import defaultdict
from os import makedirs
from os.path import exists, join
from pathlib import Path
from shutil import copy2

import numpy as np
from tqdm import tqdm

# Situation: generated 352,692 synthetic mineral spectrums
# Distribution minerals: {11: 29500, 19: 29500, 26: 29500, 28: 29500, 35: 29500, 41: 29500, 73: 29500, 80: 29453, 86: 29365, 88: 29210, 97: 29137, 98: 29027}
# Distribution configs: {0: 70575, 1: 70565, 2: 70541, 3: 70512, 4: 70499}
# will set aside 2x 15% of each mineral for eval and test, and discard 500 samples to work with even 29,000 for all

data_path = r"E:\Data\LIBSqORE\synthetic_all\results"
# one key-entry for each mineral id, a defaultdict of lists as value to collect different configurations per mineral
# result is a dictionary of filenames per mineral id and config
data = {m_id : defaultdict(list) for m_id in [11, 19, 26, 28, 35, 41, 73, 80, 86, 88, 97, 98]}
for data_file in tqdm(Path(data_path).rglob('*npy'), total=352692, unit_scale=True, desc='all  '):
    info = data_file.name.split('_')
    id = int(info[3])
    conf = int(info[6])
    data[id][conf].append(data_file)

# split up collected files evenly (minerals and configs) into train, eval and test sets
train, eva, test = list(), list(), list()
slices = [(0, 870, test, 'test '), (870, 1740, eva, 'eval '), (1740, 5800, train, 'train')]
for sl1, sl2, lst, name in slices:
    for k in tqdm(data.keys(), desc=name, unit_scale=True):
        for v in sorted(data[k].values()):
            lst.extend(v[sl1:sl2])

files_to_delete = list()
relabel_dict = {m_id : [0]*5 for m_id in [11, 19, 26, 28, 35, 41, 73, 80, 86, 88, 97, 98]}
# remove worker_thread-id designation from filename and move it into split folder
for lst, split in [(sorted(test),'test'), (sorted(eva),'eval'), (sorted(train),'train')]:
    for file_obj in tqdm(lst, desc='cp_'+split, unit_scale=True):
        # grab info to construct new filename and path
        info = str(file_obj.name).split('_')
        m_id = int(info[3])
        conf = int(info[6])
        new_nr = relabel_dict[m_id][conf]
        new_name = '{}_{:05}'.format('_'.join(info[3:-1]), int(new_nr))
        dest_dir = join('E:\Data\LIBSqORE\synthetic_all', split)

        if not exists(dest_dir):
            makedirs(dest_dir)
        dest = join(dest_dir, new_name)

        # convert to npz and store at destination
        data, labels = np.load(file_obj, allow_pickle=True)
        np.savez_compressed(dest, data=data, labels=labels)
        # copy2(str(file_obj), dest)
        relabel_dict[m_id][conf] += 1
        files_to_delete.append(str(file_obj))

with open('files_to_delete.txt', 'w') as f:
    for i in files_to_delete:
        f.write(f'{i}\n')
