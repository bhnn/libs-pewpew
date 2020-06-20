import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

with open('config/datasets.yaml') as cnf:
    dataset_configs = yaml.safe_load(cnf)

minerals_12 = [(11, 'LIBS002 LIBS006 LIBS007 LIBS008 LIBS009 LIBS010 LIBS011 LIBS012 LIBS013 LIBS014 LIBS015 LIBS016 LIBS017 LIBS018 LIBS019 LIBS020'), # azurite
            (26, 'LIBS028 LIBS029 LIBS030 LIBS031 LIBS032 LIBS033 LIBS034 LIBS035 LIBS036 LIBS037 LIBS038 LIBS039'), # chalcopyrite
            (41, 'cupri LIBS044 LIBS045 LIBS046 LIBS047 LIBS048 LIBS049 LIBS051 LIBS198 LIBS199 LIBS200 LIBS201 LIBS202 LIBS203'), # cuprite
            (73, 'LIBS059 LIBS060 LIBS061 LIBS062 LIBS063 LIBS064 LIBS065 LIBS066 LIBS067 LIBS068 LIBS069 LIBS070 LIBS071 LIBS072 LIBS073 LIBS074 LIBS075'), # malachite
            (80, 'oliveni LIBS077 LIBS078 LIBS079 LIBS135'), # olivenite
            (98, 'tetrahedr LIBS088 LIBS089'), # tetrahedrite
            (28, 'chalcotri LIBS040'), # chalcotrichite
            (97, 'tenori LIBS155'), # tenorite
            (88, 'rosasi LIBS192'), # rosasite
            (19, 'borni LIBS021 LIBS144'), # bornite
            (86, 'pseudomalachi LIBS081 LIBS082 LIBS175'), # pseudomalachite
            (35, 'corneti LIBS139') # cornetite
            ]
minerals_12_id = [a_tuple[0] for a_tuple in minerals_12]

minerals_all = [(1, 'LIBS105'), (2, 'LIBS005 LIBS119'), (3, 'LIBS103'), (4, 'LIBS121'), (5, 'LIBS148'), (6, 'LIBS107'), (7, 'LIBS101'), (8, 'LIBS104'), (9, 'LIBS106'), (10, 'LIBS166'),
            (11, 'LIBS002 LIBS006 LIBS007 LIBS008 LIBS009 LIBS010 LIBS011 LIBS012 LIBS013 LIBS014 LIBS015 LIBS016 LIBS017 LIBS018 LIBS019 LIBS020'),
            (12, 'LIBS143'), (13, 'LIBS123'), (14, 'LIBS168'), (15, 'LIBS120'), (16, 'LIBS140'), (17, 'LIBS154'), (18, 'LIBS125'), (19, 'borni LIBS021 LIBS144'),  (20, 'LIBS170'),
            (21, 'LIBS022 LIBS164'), (22, 'LIBS023 LIBS025 LIBS027 LIBS183'), (23, 'LIBS117'), (24, 'LIBS137'),
            (26, 'LIBS028 LIBS029 LIBS030 LIBS031 LIBS032 LIBS033 LIBS034 LIBS035 LIBS036 LIBS037 LIBS038 LIBS039'),
            (28, 'chalcotri LIBS040'), (29, 'LIBS153'), (30, 'LIBS041'), (31, 'LIBS115'), (32, 'LIBS159'), (33, 'LIBS162'), (34, 'LIBS126'),
            (35, 'corneti LIBS139'),  (36, 'LIBS127'), (37, 'LIBS167'), (38, 'LIBS185'), (39, 'LIBS187'), (40, 'LIBS165'),
            (41, 'cupri LIBS044 LIBS045 LIBS046 LIBS047 LIBS048 LIBS049 LIBS051 LIBS198 LIBS199 LIBS200 LIBS201 LIBS202 LIBS203'), # cuprite
            (42, 'LIBS128'), (43, 'LIBS145'), (44, 'LIBS118'), (45, 'LIBS055'), (46, 'LIBS097'), (47, 'LIBS161'), (49, 'LIBS099'), (50, 'LIBS138'), (52, 'LIBS174'),
            (53, 'LIBS172'), (54, 'LIBS136'), (55, 'LIBS109'), (56, 'LIBS152'), (57, 'LIBS188'), (58, 'LIBS163'), (59, 'LIBS191'), (60, 'LIBS122'),
            (61, 'LIBS150'), (62, 'LIBS094'), (63, 'LIBS151'), (64, 'LIBS173'), (65, 'LIBS056 LIBS180'), (66, 'LIBS116'), (67, 'LIBS182'), (69, 'LIBS057 LIBS058 LIBS171'),
            (70, 'LIBS189'), (71, 'LIBS130'), (72, 'LIBS133'), (73, 'LIBS059 LIBS060 LIBS061 LIBS062 LIBS063 LIBS064 LIBS065 LIBS066 LIBS067 LIBS068 LIBS069 LIBS070 LIBS071 LIBS072 LIBS073 LIBS074 LIBS075'), (74, 'LIBS110'),
            (75, 'LIBS179'), (76, 'LIBS076 LIBS114'), (77, 'LIBS193'), (78, 'LIBS156'), (79, 'LIBS190'),
            (80, 'oliveni LIBS077 LIBS078 LIBS079 LIBS135'), (81, 'LIBS157'), (82, 'LIBS146'), (84, 'LIBS177'), (85, 'LIBS142'), (86, 'pseudomalachi LIBS081 LIBS082 LIBS175'), (87, 'LIBS132'),
            (88, 'rosasi LIBS192'),  (89, 'LIBS096'), (90, 'LIBS084 LIBS134'), (91, 'LIBS131'), (92, 'LIBS085 LIBS086 LIBS100 LIBS100'), (94, 'LIBS149'), (95, 'LIBS087'),
            (96, 'LIBS169'), (97, 'tenori LIBS155'), (98, 'tetrahedr LIBS088 LIBS089'),  (99, 'LIBS102'), (100, 'LIBS091'), (101, 'LIBS184'), (102, 'LIBS181'), (103, 'LIBS124'), (104, 'LIBS092 LIBS147'), (105, 'LIBS129'), (106, 'LIBS098'), (107, 'LIBS093')
        ]
minerals_all_id = [a_tuple[0] for a_tuple in minerals_all]


def find_min_max(mineralid, datasetpath, datasetname):
    """
    Finds min and max number of spectra per mineral in dataset
    :param mineralid:   mineralids of the dataset
    :param datasetpath: path to the dataset
    :param datasetname: datasetname for visualisation
    :returns:           saves visualisation in data/visualisations/datadistribution as pdf and png
    """

    num_spec = []
    for m_id in mineralid:
        m_id = '{0:04d}'.format(m_id)
        files = sorted(glob.glob(os.path.join(datasetpath,'train',m_id + '*.npz')))
        files.extend(sorted(glob.glob(os.path.join(datasetpath,'test', m_id+'*.npz'))))
        num_spec.append(len(files))

    print(datasetname)
    print('Minimum number of spectra per mineral: ',np.min(num_spec))
    print('Maximum number of spectra per mineral: ',np.max(num_spec))
    print('Number of minerals with more than 1000 spectra: ',sum([i>1000 for i in num_spec]))
    print('Number of minerals with less than 1000 spectra: ', sum([i<1000 for i in num_spec]))

    plt.hist(num_spec, bins=30, color='#0d627a')
    plt.title('Distribution of the dataset: '+ datasetname, fontsize=12)
    plt.xlabel('Number of spectra', fontsize=12)
    plt.ylabel('Number of minerals', fontsize=12)
    plt.savefig(os.path.join(dataset_configs['repo_path'], 'data/visualisations/datadistribution.pdf'))
    plt.savefig(os.path.join(dataset_configs['repo_path'], 'data/visualisations/datadistribution.png'))

#find_min_max(minerals_12_id, dataset_configs['hh_12_path']  , dataset_configs['hh_12_str'])
find_min_max(minerals_all_id, dataset_configs['hh_all_path'], dataset_configs['hh_all_str'])
