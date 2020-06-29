import os
from pathlib import Path
from shutil import copy2
import sys
import numpy as np
from tqdm import tqdm
import yaml


def organise_datasets(output_path, repo_path, raw_path, minerals, datasetname):
    """
    This script takes all the measurements for the given list of minerals and
    copies them from individual folders in the old file system into the new
    output directory.
    Saves files in the format: Mineral-ID_Class_Subgroup_Measurepoint_Shot.csv

    :param repo_path:   path to repository
    :param raw_path:    path to folder containing the raw dataset provided by Pia Brinkmann
    :param output_path: path for the new handheld dataset
    :minerals:          list containing tuples of (mineral ID, all foldernames for this mineral)
    :datasetname:       name of the dataset for the logfile
    :returns:           Saves files in the format: Mineral-ID_Class_Subgroup_Measurepoint_Shot.csv in the output directory
    """


    # information on all minerals to include mineral class and mineral group
    minerals_file = os.path.join(repo_path,'data/synthetic_minerals.npy')
    all_list = np.load(minerals_file, allow_pickle=True)
    # globbed filenames of all .csv files in the data directory and subdirectories
    for mineral_id, mineral_code in tqdm(minerals):
        codes = mineral_code.split(' ')
        filepaths = list()
        # iterate over entire directory structure to filter out spectra that belong to the current mineral
        for f in Path(raw_path).rglob('*.[cC][sS][vV]'):
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
        mineral_id = int(mineral_id)

        mineral_class = all_list[mineral_id][4]
        mineral_subgroup = all_list[mineral_id][5]
        output_counter = 0

        # create output directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(raw_path, 'transformation_log_'+datasetname+'.txt'), 'a') as logfile:
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


if __name__ == '__main__':
    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
    try:
        hh_12_path = dataset_configs['hh_12_path']
        hh_12_name = dataset_configs['hh_12_name']

        hh_all_path = dataset_configs['hh_all_path']
        hh_all_name = dataset_configs['hh_all_name']

        repo_path = dataset_configs['repo_path']
        raw_path = dataset_configs['raw_path']

    except KeyError as e:
        print(f'Missing dataset config key: {e}')
        sys.exit(1)

    minerals_all = np.load(os.path.join(repo_path, 'data/mineral_infos/mineral_id_folder_100.npy'))
    minerals_12 = np.load(os.path.join(repo_path, 'data/mineral_infos/mineral_id_folder_12.npy'))
    organise_datasets(hh_all_path, repo_path, raw_path, minerals_all, hh_all_name)
    organise_datasets(hh_12_path, repo_path, raw_path, minerals_12, hh_12_name)
