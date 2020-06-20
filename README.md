# libs-pewpew

This Project aims to classify mineral classes, mineral groups and minerals.
The dataset consists of measurements using laser induced breakdown spectroscopy of 100 minerals.
A synthetic dataset can be generated with the code below to evaluate a potential increase in accuracy with the additionally generated data.

- change paths in config/datasets.yaml

#### Generate synthetic dataset:

To complement the handheld dataset, a synthetic dataset can be generated to support the given but rather small dataset. It is based on the atomic composition of the elements of 12 minerals (see data/synthetic_minerals_raw.csv).

- Code to generate this


#### Prepare the synthetic dataset:
The synthetic dataset has to be prepared to be useful for training an ANN.

- unzip .tar files
- run prepare_all_synthetic.py
- make sure there are the folders synthetic_all/train, test, eval
- run npy_to_npz.py

#### Handheld dataset:
The handheld dataset has to be prepared to be useful for training an ANN, as it is currently in different folders etc.

Organise handheld dataset in two folders: hh_12 and hh_all, Saves files in the format: Mineral-ID_Class_Subgroup_Measurepoint_Shot.csv
- python3 src/handheld_dataset/organise_hh_dataset.py
Convert csv files to npz files, train test split while not seperating measure points
- csv to numpy (also train, test split)
As part of the preprocessing: baseline correction of all spectra (to save space, delete old folders)
- python3 src/handheld_dataset/baselinecorrection.py (expects the folders "train_uncorrected" and "test_uncorrected")


### Artificial Neural Networks:

#### Mixture of handheld and synthetic dataset with baseline MLP:
To evaluate the influence of the generated synthetic data on the accuracy of the trained ANN, the following code will train a baseline MLP with a mixture of the handheld dataset with the synthetic dataset from 100% handheld + 0% synthetic to 100 % handheld + 100% synthetic.

- python3 src/mixture_mlp.py -r 5 -e 10  -c 0
- python3 src/mixture_mlp.py -r 5 -e 10  -c 1
- python3 src/mixture_mlp.py -r 5 -e 10  -c 2

The results can be visualized:
- python3 src/visualisation/plot_mixture_results.py

#### Comparison of preprocessing methods with baseline MLP:

- change -c for three classification targets and -n for three normalisation methods
- 3 repetitions, 5 epochs, handheld dataset with 12 minerals + synthetic dataset with 12 minerals
- python 3 baseline_mlp.py -r 3 -e 5 -d 2 -c 0/1/2 -n 0/1/2


### Visualizations

All visualisations will be saves in libs-pewpew/data/visualisations

#### Plot average Spectra

Calculates and saves average synthetic and average handheld spectra for the 12 minerals:
- python3 src/visualisation/average_spectra.py

To plot the handheld dataset:
- python3 src/visualisation/plot_average_spectra_hh12.py

To plot the comparison of handheld and synthetic data:
- python3 src/visualisation/plot_syn_hh_spectra.py

#### Plot baseline correction

Visualisation of uncorrected spectrum, the baseline and the corrected spectrum

- python3 src/visualisation/plot_baseline.py


#### Plot results of the mixture of datasets

- python3 src/visualisation/plot_mixture_results.py

#### Minimum and maximum amount of spectra in the dataset

-python3 src/visualisation/minmaxdataset.py




TODO:
- handheld_dataset: organise_hh_dataset change paths
- handheld dataset: delete "organise all minerals"
- bundle numpy: csvtonumpy muss in handheld order und restliche files löschen oder?
- was machen die files in "numpy_minerals"?
- ordner "synthetic_dataset" muss nochmal durchgegangen werden
- welche files können im src weg? Welche models haben wir nicht benutzt? Andere files auch nochmal überprüfen
