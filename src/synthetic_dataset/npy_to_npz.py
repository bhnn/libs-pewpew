import numpy as np
import glob
import os
import tqdm

def convert_to_npz(path):
    files = sorted(glob.glob(os.path.join(path,'*.npy')))
    for f in files:
        data = np.load(f, allow_pickle=True)
        filename = f[:-4] # filename without .npy
        mineral_id, mineral_class, mineral_subgroup, _, _ = f.split('/')[-1].split('_')
        labels = np.asarray((mineral_class, mineral_subgroup, mineral_id)).astype(int)
        np.savez_compressed(filename, data=data, labels=labels)
        os.remove(f) #delete npy file

convert_to_npz('/samba/cjh/julia/synthetic_all/train')
convert_to_npz('/samba/cjh/julia/synthetic_all/test')
convert_to_npz('/samba/cjh/julia/synthetic_all/eval')
