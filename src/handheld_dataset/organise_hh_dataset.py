import os
from pathlib import Path
from shutil import copy2

import numpy as np

minerals = [(11, 'LIBS002 LIBS006 LIBS007 LIBS008 LIBS009 LIBS010 LIBS011 LIBS012 LIBS013 LIBS014 LIBS015 LIBS016 LIBS017 LIBS018 LIBS019 LIBS020'), # azurite
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

# file locations
minerals_file = r'D:\Dropbox\uni\3_semester\ml\libs-pewpew\data\synthetic_minerals.npy'
hh_data = r'E:/Data/ML Data'
output_path = os.path.join(hh_data, 'pretty_data')

# information on all minerals
all_list = np.load(minerals_file, allow_pickle=True)
# globbed filenames of all .csv files in the data directory and subdirectories

for mineral_id, mineral_code in minerals:
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
    print('Found', len(filepaths),'spectra for mineral', all_list[mineral_id][1])
    unique_names = list()
    counter = -1
    sorted_by_parent = list()
    # sorts all files by path alphabetically, .csv files are out of order with Shot(59), Shot(6), Shot(61) but we only
    # care about the names of their parent directories anyway
    for f in sorted(filepaths):
        # if a new parent directory is traversed, start adding the following .csvs to a new sublist
        if f.parent not in unique_names:
            unique_names.append(f.parent)
            counter += 1
            sorted_by_parent.append(list())
        else:
            sorted_by_parent[counter].append(f)
    
    mineral_class = all_list[mineral_id][4]
    mineral_subgroup = all_list[mineral_id][5]
    output_counter = 0

    # create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(hh_data, 'transformation_log.txt'), 'a') as logfile:
        for directory in sorted_by_parent:
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
            # {mineral_id}_{mineral_class}_{mineral_subgroup}_{index of original directory}_{running index}.csv
            mineral_counter = 0
            for f in directory:
                new_file_name = '{:04}_{:03}_{:03}_{:03}_{:05}.csv'.format( mineral_id, 
                                                                            mineral_class, 
                                                                            mineral_subgroup, 
                                                                            output_counter, 
                                                                            mineral_counter)
                copy2(str(f), os.path.join(output_path, new_file_name))
                mineral_counter += 1
            output_counter += 1
    print('Finished with mineral:', all_list[mineral_id][1])
