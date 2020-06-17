import yaml
from glob import glob
from os.path import join
from utils import __get_labels
from numpy import unique, append, mean, std, min, max, sort

# calculates the amount of samples per class for HH12 and HH100, with standard deviation and top5 min/max values
for name, dataset in [('HH12', 'hh_12_path'), ('HH100', 'hh_all_path')]:
    with open('config/datasets.yaml') as cnf:
        # grab file paths
        dataset_configs = yaml.safe_load(cnf)
        data_path = dataset_configs[dataset]

    train_data = sorted(glob(join(data_path, 'train', '*.npz')))
    test_data = sorted(glob(join(data_path, 'test', '*.npz')))

    # read labels from filenames, faster than running the generator
    train_labels = __get_labels(train_data, 2)
    test_labels = __get_labels(test_data, 2)
    labels = append(train_labels, test_labels)

    # count number of unique occurrences for each label in the data, then calculate mean, std, bottom/top 5
    counts = unique(labels, return_counts=True)[1]
    print(f'{name}:\taverage of {round(mean(counts),0)} samples per class, +/- {round(std(counts),1)} stddev\t(min: {sort(counts)[:5]}, max: {sort(counts)[::-1][:5]})')