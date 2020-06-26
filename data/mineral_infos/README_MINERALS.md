### README for files in this folder

##### This folder contains the following files:

###### ZusammenfassungMinerale.xlsx:
This file is provided by Pia Brinkmann (University of Potsdam, pbrinkmann@uni-potsdam.de) and contains detailed information on the minerals used for this project. Most importantly one can find the foldernames for each mineral here.
The dataset provided by Pia Brinkmann contains folders in the format "LIBSXXX" and each folder contains LIBS measurements for a mineral.


###### synthetic_minerals_raw.csv

This is a handwritten file that contains the following information:

- Mineral ID(1-108), Mineral Name, Elements of the mineral, % for each element, Mineral class name, Mineral group name
- Example: 2,aikinite,"[S, Cu, Pb, Bi]","[50.00, 16.67, 16.67, 16.67]", sulfides, tetrahedrite-bournonite-boulangerite

The Elements and elemental composition is taken from: https://www.mineralienatlas.de/

The mineral class and mineral group is taken from: ZusammenfassungMinerale.xlsx

This file is saved as a numpy file using libs-pewpew/src/synthetic_dataset/prepare_source_list.py and saved in libs-pewpew/data/synthetic_minerals.npy with mineral ID (starting from 0-107 here) a list of elements, a list of % of those elements and the mineral class number and the mineral group number.
- Example: [1 'aikinite' list(['S', 'Cu', 'Pb', 'Bi']) array([50. , 16.67, 16.67, 16.67]) 6 1]

The mineral IDs in this file (data/synthetic_minerals.npy) and the official ones used for mineral, mineral class and mineral group!!!

##### number_classes_groups.txt

This file tell you which mineral class name and mineral group name corresponds to which number.
Mainly for understanding classification results (some classes are more similar than other and might be more difficult to learn)


#### organise_all_minerals.py

This file contains handwritten lists with the mineral ID based on the file "data/synthetic_minerals.npy" and the foldernames in Pia's file system taken from "ZusammenfassungMinerale.xlsx".
These lists are used to organise the data into useful folders and by running this file, the folders "hh_100" and "hh_12" are filled with the right data and the filenames contain mineral ID, mineral class, mineral group etc.


#### Additional data in the future:

If there are additional measurements in the future with completely new minerals,
one has to:
- update the file "data/mineral_infos/synthetic_minerals_raw.csv"
- run python src/synthetic_dataset/prepare_source_list.py to update the file "data/synthetic_minerals.npy"
- update the lists in "data/mineral_infos/organise_all_minerals.py" and run this file
- run src/handheld_dataset/organise_hh_dataset

If there are only additional measurements for minerals that already exists, one only has to:
- update the lists in "data/mineral_infos/organise_all_minerals.py" with the additional foldernames and run this file again
- run src/handheld_dataset/organise_hh_dataset
