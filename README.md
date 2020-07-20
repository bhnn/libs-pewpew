# Classify minerals using lasers

The goal of this project is to classify spectral data of copper-based minerals into mineral classes, mineral subgroups and minerals. Gold-labels used for classification are based on the book *Lehrbuch der Mineralogie* (Rösler, 1985). The dataset consists of measurements taken using laser-induced breakdown spectroscopy (LIBS) of 100 distinct minerals. Additionally, synthetic spectral data can be generated using the code below to augment the size of the real-world data and potentially improve classification accuracy.

## Getting started

### Prerequisites

Access to the dataset of handheld LIBS measurements can be requested from Dr. Daniel Riebe at the Department of Physical Chemistry, university of Potsdam.

### Installing

Fork and clone the repository to your drive, create a clean virtual environment **(Python 3.6)** with a method of your choosing, then enter the cloned directory and run

```pip install -r requirements.txt```

to install all packages required to run the code of this project. Then, the file [config/datasets.yaml](config/datasets.yaml) must be edited to include absolute paths to your cloned repository as well as the parent directory of the unprocessed handheld (and any possible synthetic) dataset (`raw_path`). The paths of the preprocessed datasets (two handheld, one synthetic) need to be added later as well.

### Real-world LIBS data
After accessing the handheld dataset, several preparations need to be completed before it can be used with models. 

First, the relevant mineral spectra must be extracted from the large collection of `.csv` files spread out across many nested subdirectories. Running

```python src/handheld_dataset/organise_hh_dataset.py```

will collate a 100-mineral dataset (**HH100**) and a 12-mineral subset (**HH12**) and organise them in separate folders each. Afterwards, they can be converted to compressed numpy (`.npz`) files using 

```python src/handheld_dataset/csvtonumpy.py```

as well as split them into training and test sets while respecting measure point boundaries. The naming convention for the data files conforms to `<mineral id>_<class id>_<subgroup id>_<measure point #>_<shot #>.npz`

Finally, baseline correction needs to be applied to adjust for measurement noise artefacts:

```python src/handheld_dataset/baselinecorrection.py```

To save disk space, the unprocessed `.csv` data files as well as the uncorrected dataset directories can be deleted, as baseline correction creates duplicates of the datasets.

### Generate synthetic dataset:

Because the handheld dataset is of limited size, synthetic data can be generated to inflate it and theoretically bolster classification performance. In our testing, 12 minerals were selected for this that were most prevalent in the handheld data and spanned 4 of the most important mineral classes. Detailed information about the gold-standard labels and atomic compositions of all minerals in both handheld and synthetic data can be found in [data/mineral_info/synthetic_minerals_raw.csv](../libs-backup/data/synthetic_minerals_raw.csv).

First, the above-mentioned mineral information needs to be encoded into numeric values that serve as the labels for generated data. Running

```python src/synthetic_dataset/prepare_source_list.py```

will accomplish that, as well as run several checks to assert validity of the mineral data. Afterwards, the generation process can be started using

```python src/synthetic_dataset/create_synthetic_np.py -p <..> -i <..>```

with `-p` being the number of threads that can be used and `-i` being a manually picked worker-id prepended to finished data to guarantee unique file names. This process is an **endless loop** and thus needs to be stopped manually. 

Place the generated synthetic data in a subdirectory named `synthetic` in the `raw_path` location defined in the config earlier. Now, run

```python src/synthetic_dataset/prepare_all_synthetic.py```

to split the data into train, eval and test sets (default: 70/15/15) and convert the files to `.npz` format.

### Test the setup

To test whether everything is set up correctly, try running one of the models:

```python src/baseline_mlp.py -e 1 -c 2 -d 1```

This should start a simple model to examine a performance baseline, for 1 epoch (`-e`) with no repetitions on the 12-mineral subset.

For a detailed explanation of command line arguments, start any model with the `-h` parameter.

---

## Running different models

These models are available to test different hypotheses:
* [64Shot](src/64shot_mlp.py)
* [Baseline](src/baseline_mlp.py)
* [Concatenation](src/concat_mlp.py)
* [Decision Tree](src/decision_tree.py)
* [Mixture](src/mixture_mlp.py)
* [Transfer DT](src/transfer_dt.py)
* [Transfer MLP](src/transfer_mlp.py)

### Baseline model

All models (except decision trees) are either using all of the baseline architecture or slightly modified parts of it. This allows them to be compared with this baseline model more easily.

#### Example: Performance impact of preprocessing choices

```python src/baseline_mlp.py -r 3 -e 10 -c 2 -d 1 -n <{0, 1, 2}>```

This would run the baseline model 3 times for 10 epochs each on the HH12 dataset with minerals as the classification target using the chosen normalisation method: `0` for *no* normalisation, `1` for *standard normal variate* normalisation or `2` for *min-max* normalisation. Prediction accuracy results of models with a `-r` value > 1 will be averaged over all model repetitions. These results can then be compared with those from the other normalisation methods.

### Heatmap Paradigm (64Shot)

Hypothesis: LIBS device measurements are taken as 64 shots in an 8x8 grid, presumed to be aimed at the centre of minerals by the conducting researcher. Thus, classification accuracy in the centre of the grid should be higher than at the edges of the grid.

```python src/64shot_mlp.py -e 10 -c 2 -d 1```

This trains a baseline model for 10 epochs on the HH12 dataset with minerals as the classification target. Afterwards, the model arranges the test set's measure points into its original 8x8 grids and predicts accuracies for each shot in the grid. All intermediate heatmap results for measure points are saved to the `results/` folder and the result of calculating an average across all measure point heatmaps is displayed.

### Transfer learning & concatenation

There are 2 transfer learning concepts available: classic transfer learning where the final layer is replaced for fine-tuning on another dataset, and [concatenated transfer learning](https://i.imgur.com/ARHIcfq.png) where the output of the last dense layer of one network is concatenated to the input of another network that is fine-tuned on the secondary dataset.

For these models, no dataset can be selected with command line parameters, as they only work with a combination of the HH12 dataset and a synthetic dataset containing the same 12 minerals.

```python src/concat_mlp.py -r 3 -e 10 -c 2```

```python src/transfer_mlp.py -r 3 -e 10 -c 2```

### Performance influence of mixture of handheld and synthetic dataset

To evaluate the influence of the generated synthetic data on the accuracy of neural network training, the following code will train a baseline model with a mixture of HH12 and the synthetic dataset: from a composition 100% handheld + 0% synthetic data to 100 % handheld + 100% synthetic data.

```python src/mixture_mlp.py -r 5 -e 10 -c 2```

---

## Visualizations

All visualisations will be saves in *data/visualisations*.

### Plot average spectra per mineral

Calculates and saves average synthetic and average handheld spectra for the 12 minerals:

```python src/visualisation/average_spectra.py```

To plot the handheld dataset:

```python src/visualisation/plot_average_spectra_hh12.py```

To plot the comparison of handheld and synthetic data:

```python src/visualisation/plot_syn_hh_spectra.py```

### Plot baseline correction

Visualisation of uncorrected spectrum, the baseline and the corrected spectrum

``` python3 src/visualisation/plot_baseline.py```

### Plot results of the mixture of datasets

```python3 src/visualisation/plot_mixture_results.py```

### Minimum and maximum amount of spectra in the dataset

``` python3 src/visualisation/minmaxdataset.py```

## Acknowledgements
Thanks to
* Pia Brinkmann, Dr. Daniel Riebe and the Department of Physical Chemistry for providing the dataset of handheld LIBS measurements
* Federico Malerba for providing parallelisable code to generate interpolated LIBS spectra from stick plot data
* Dr. Paul Prasse for providing code to handle receiving spectral data from the [NIST LIBS database](https://physics.nist.gov/PhysRefData/ASD/LIBS/libs-form.html)