import os
from pathlib import Path
from shutil import copy2

import numpy as np
from tqdm import tqdm

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

minerals_6 = [(11, 'LIBS002 LIBS006 LIBS007 LIBS008 LIBS009 LIBS010 LIBS011 LIBS012 LIBS013 LIBS014 LIBS015 LIBS016 LIBS017 LIBS018 LIBS019 LIBS020'), # azurite
                 (22, 'LIBS023 LIBS025 LIBS027 LIBS183'),
                 (26, 'LIBS028 LIBS029 LIBS030 LIBS031 LIBS032 LIBS033 LIBS034 LIBS035 LIBS036 LIBS037 LIBS038 LIBS039'), # chalcopyrite
                 (30, 'LIBS041'), (73, 'LIBS059 LIBS060 LIBS061 LIBS062 LIBS063 LIBS064 LIBS065 LIBS066 LIBS067 LIBS068 LIBS069 LIBS070 LIBS071 LIBS072 LIBS073 LIBS074 LIBS075'), # malachite
                 (98, 'tetrahedr LIBS088 LIBS089'), # tetrahedrite
                 ]

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
            (80, 'oliveni LIBS077 LIBS078 LIBS079 LIBS135'), (81, 'LIBS157'), (82, 'LIBS146'), (84, 'LIBS177'), (85, 'LIBS142'), (86, 'pseudomalachi LIBS081 LIBS082 LIBS175'),  (87, 'LIBS132'),
            (88, 'rosasi LIBS192'),  (89, 'LIBS096'), (90, 'LIBS084 LIBS134'), (91, 'LIBS131'), (92, 'LIBS085 LIBS086 LIBS100 LIBS100'), (94, 'LIBS149'), (95, 'LIBS087'),
            (96, 'LIBS169'), (97, 'tenori LIBS155'), (98, 'tetrahedr LIBS088 LIBS089'),  (99, 'LIBS102'), (100, 'LIBS091'), (101, 'LIBS184'), (102, 'LIBS181'), (103, 'LIBS124'), (104, 'LIBS092 LIBS147'), (105, 'LIBS129'), (106, 'LIBS098'), (107, 'LIBS093')
        ]

def organise_datasets(minerals, outputname):
    """
    Saves files in the format:
    Mineral-ID_Class_Subgroup_Measurepoint_Shot.csv
    """
    # file locations
    minerals_file = r'data/synthetic_minerals.npy'
    hh_data = r'/media/ben/Volume/ml_data/hh_raw'
    output_path = os.path.join(hh_data, outputname)

    # information on all minerals
    all_list = np.load(minerals_file, allow_pickle=True)
    # globbed filenames of all .csv files in the data directory and subdirectories

    for mineral_id, mineral_code in tqdm(minerals):
        codes = mineral_code.split(' ')
        filepaths = list()
        # iterate over entire directory structure to filter out spectra that belong to the current mineral
        for f in Path(hh_data).rglob('*.[cC][sS][vV]'):
            # filename and codes need lowercase for comparison, e.g. to make finding 'azurit' in 'Azurite' work
            f_str = str(f).lower()
            # ignore these 2 folders
            if 'cu2s_in_basalt' not in f_str and 'cufes2_in_basalt' not in f_str:
                # check all mineral codes at once
                for c in codes:
                    if c.lower() in f_str:
                        filepaths.append(f)
        unique_names = list()
        counter = -1
        sorted_by_parent = list()
        # sorts all files by path alphabetically, .csv files are out of order with Shot(59), Shot(6), Shot(61) but we only
        # care about the names of their parent directories anyway
        for f in sorted(filepaths):
            if not f.name.startswith('.'): #exclude hidden files
                # if a new parent directory is traversed, start adding the following .csvs to a new sublist
                if f.parent not in unique_names:
                    unique_names.append(f.parent)
                    counter += 1
                    sorted_by_parent.append(list())
                sorted_by_parent[counter].append(f)

        mineral_class = all_list[mineral_id][4]
        mineral_subgroup = all_list[mineral_id][5]
        output_counter = 0

        # create output directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(hh_data, 'transformation_log_'+outputname+'.txt'), 'a') as logfile:
            for directory in tqdm(sorted_by_parent, leave=False):
                # write log file entry to mark the transformation of directory names to numeric indices in case anyone needs
                # this information later
                dir_name = str(directory[0].parent)[15:]
                log_entry = '\"{}\" transformed to {:04}_{:03}_{:03}_{:03}_XXXXX\n'.format( dir_name,
                                                                                            mineral_id,
                                                                                            mineral_class,
                                                                                            mineral_subgroup,
                                                                                            output_counter)
                logfile.write(log_entry)

                # go through list of .csv files and copy them to output directory with a new name:
                # {mineral_id}_{mineral_class}_{mineral_subgroup}_{index of original directory}_{shot id}.csv
                mineral_counter = 0
                for f in directory:
                    # use original id to preserve integrity of data for later
                    shot_id = int(f.name[5:-5])
                    new_file_name = '{:04}_{:03}_{:03}_{:03}_{:05}.csv'.format( mineral_id,
                                                                                mineral_class,
                                                                                mineral_subgroup,
                                                                                output_counter,
                                                                                shot_id)
                    copy2(str(f), os.path.join(output_path, new_file_name))
                output_counter += 1

organise_datasets(minerals=minerals_all, outputname = 'hh_all')
organise_datasets(minerals=minerals_6, outputname = 'hh_6')
organise_datasets(minerals=minerals_12, outputname = 'hh_12')
