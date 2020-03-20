import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
from tqdm import tqdm
import seaborn as sns


path = r'/Volumes/Samsung_T5/LIBSqORE_Austausch'

minerals_6 = [(11, 'LIBS002 LIBS006 LIBS007 LIBS008 LIBS009 LIBS010 LIBS011 LIBS012 LIBS013 LIBS014 LIBS015 LIBS016 LIBS017 LIBS018 LIBS019 LIBS020'), # azurite
                 (22, 'LIBS023 LIBS025 LIBS027 LIBS183'),
                 (26, 'LIBS028 LIBS029 LIBS030 LIBS031 LIBS032 LIBS033 LIBS034 LIBS035 LIBS036 LIBS037 LIBS038 LIBS039'), # chalcopyrite
                 (30, 'LIBS041'),
                 (73, 'LIBS059 LIBS060 LIBS061 LIBS062 LIBS063 LIBS064 LIBS065 LIBS066 LIBS067 LIBS068 LIBS069 LIBS070 LIBS071 LIBS072 LIBS073 LIBS074 LIBS075'), # malachite
                 (98, 'tetrahedr LIBS088 LIBS089'), # tetrahedrite
                 ]
minerals_6_id = [a_tuple[0] for a_tuple in minerals_6]

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
            (80, 'oliveni LIBS077 LIBS078 LIBS079 LIBS135'), (81, 'LIBS157'), (82, 'LIBS146'), (84, 'LIBS177'), (85, 'LIBS142'), (86, 'pseudomalachi LIBS081 LIBS082 LIBS175'),  (87, 'LIBS132'),
            (88, 'rosasi LIBS192'),  (89, 'LIBS096'), (90, 'LIBS084 LIBS134'), (91, 'LIBS131'), (92, 'LIBS085 LIBS086 LIBS100 LIBS100'), (94, 'LIBS149'), (95, 'LIBS087'),
            (96, 'LIBS169'), (97, 'tenori LIBS155'), (98, 'tetrahedr LIBS088 LIBS089'),  (99, 'LIBS102'), (100, 'LIBS091'), (101, 'LIBS184'), (102, 'LIBS181'), (103, 'LIBS124'), (104, 'LIBS092 LIBS147'), (105, 'LIBS129'), (106, 'LIBS098'), (107, 'LIBS093')
        ]
minerals_all_id = [a_tuple[0] for a_tuple in minerals_all]

# adapted from Federico
def normalise_minmax(np_data):
    result = np.zeros((np_data.shape[0], np_data.shape[1]))
    for i in range(np_data.shape[0]):
        if np.max(np_data[i,:]) > 0:
            result[i] = np_data[i][:,1] / np.max(np_data[i][:,1])
        else:
            raise ValueError
    return result


def snv(input_data):
    # Define a new array and populate it with the corrected data
    print(input_data.shape)
    #input_data = [0 if i < 0 else i for i in input_data]
    data_snv = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        input_data[i,:] = input_data[i,:].clip(min=0)
        data_snv[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return data_snv

def PCA_analysis(dataset_id, datasetname, average=True, plot = True, label_num=0, TwoD=True):
    """
    label_num: 0 = class, 1 = subgroup, 2 = mineral

    """

    if average == True:
        all_samples = []
        labels = []
        for m_id in dataset_id:
            m_id = '{0:04d}'.format(m_id)
            samples =[]
            files = sorted(glob.glob(os.path.join(path, datasetname,'train_data', m_id+'*.npz')))
            files.extend(sorted(glob.glob(os.path.join(path, datasetname,'test_data', m_id+'*.npz'))))
            with np.load(files[0]) as npz_file_label:
                labels.append(npz_file_label['labels'][label_num])
            for f in tqdm(files):
                with np.load(f) as npz_file:
                    data = npz_file['data'][:,1] #data without wavelength
                    #data = data / np.max(data) # normalisation
                    #data = data.clip(min=0) #remove values < 0
                    samples.append(data)
                    if len(samples) > 1:
                        samples = np.mean(samples, axis=0)
                        samples = [list(samples)]

            all_samples.append(samples[0])
    else:
        all_samples = []
        labels = []
        for m_id in dataset_id:
            m_id = '{0:04d}'.format(m_id)
            files = sorted(glob.glob(os.path.join(path, datasetname,'train_data', m_id+'*.npz')))
            files.extend(sorted(glob.glob(os.path.join(path, datasetname,'test_data', m_id+'*.npz'))))
            for f in tqdm(files):
                with np.load(f) as npz_file:
                    data = npz_file['data'][:,1] #data without wavelength
                    label = npz_file['labels'][label_num]
                    #data = data / np.max(data) #normalise
                    all_samples.append(data)
                    labels.append(label)


    all_samples = np.array(all_samples)
    all_samples = snv(all_samples)
    print(all_samples.shape)


    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(all_samples)
    df_PCA = pd.DataFrame(data = principalComponents
                 , columns = ['PCA0', 'PCA1', 'PCA2'])

    if TwoD == True:
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(all_samples)
        df_PCA = pd.DataFrame(data = principalComponents
                     , columns = ['PCA0', 'PCA1'])
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="PCA0", y="PCA1",
            hue=labels,
            palette=sns.color_palette("hls", len(np.unique(labels))),
            data=df_PCA,
            legend="full",
            alpha=0.3
        )
        plt.show()
    else:
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(all_samples)
        df_PCA = pd.DataFrame(data = principalComponents
                     , columns = ['PCA0', 'PCA1', 'PCA2'])
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(
        xs=df_PCA["PCA0"],
        ys=df_PCA["PCA1"],
        zs=df_PCA["PCA2"],
        c=labels,
        cmap='tab20'
        )
        #ax.legend(handles='tab20', labels=labels)
        ax.set_xlabel('PCA0')
        ax.set_ylabel('PCA1')
        ax.set_zlabel('PCA2')
        plt.show()



PCA_analysis(dataset_id=minerals_6_id, datasetname='hh_6', average=False, label_num = 2, TwoD=True) #alle 6 minerals in 2D
#PCA_analysis(dataset_id=minerals_6_id, datasetname='hh_6', average=True, label_num = 2, TwoD=True) #average von den 6 minerals in 2D
#PCA_analysis(dataset_id=minerals_6_id, datasetname='hh_6', average=False, label_num = 2, TwoD=False) #alle 6 minerals in 3D
