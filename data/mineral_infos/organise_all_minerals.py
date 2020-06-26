import numpy as np

"""
These lists are handwritten and based on the file
- "data/mineral_infos/ZusammenfassungMinerale.xlsx" (provided by Pia Brinkmann,
University of Potsdam pbrinkmann@uni-potsdam.de) where you can find the foldernames for each mineral
- and the file "data/synthetic_minerals" for mineral IDS.


For each mineral, there are corresponding foldernames in Pia's file system.
If there are additional measurements in the future, one has to update these lists.
By running this file, the following lists will be saves as numpy files as they
are used for all kinds of operations.

This list contains all minerals with useful measurements.
Not all mineral IDS are used in this list as for some minerals there are no
measurements or the measurements are a mix of two minerals which we decided to exclude.

"""

minerals_all = [(1, 'LIBS105'),
            (2, 'LIBS005 LIBS119'),
            (3, 'LIBS103'),
            (4, 'LIBS121'),
            (5, 'LIBS148'),
            (6, 'LIBS107'),
            (7, 'LIBS101'),
            (8, 'LIBS104'),
            (9, 'LIBS106'),
            (10, 'LIBS166'),
            (11, 'LIBS002 LIBS006 LIBS007 LIBS008 LIBS009 LIBS010 LIBS011 LIBS012 LIBS013 LIBS014 LIBS015 LIBS016 LIBS017 LIBS018 LIBS019 LIBS020'), # azurite
            (12, 'LIBS143'),
            (13, 'LIBS123'),
            (14, 'LIBS168'),
            (15, 'LIBS120'),
            (16, 'LIBS140'),
            (17, 'LIBS154'),
            (18, 'LIBS125'),
            (19, 'borni LIBS021 LIBS144'), # bornite
            (20, 'LIBS170'),
            (21, 'LIBS022 LIBS164'),
            (22, 'LIBS023 LIBS025 LIBS027 LIBS183'),
            (23, 'LIBS117'),
            (24, 'LIBS137'),
            (26, 'LIBS028 LIBS029 LIBS030 LIBS031 LIBS032 LIBS033 LIBS034 LIBS035 LIBS036 LIBS037 LIBS038 LIBS039'), # chalcopyrite
            (28, 'chalcotri LIBS040'), # chalcotrichite
            (29, 'LIBS153'),
            (30, 'LIBS041'),
            (31, 'LIBS115'),
            (32, 'LIBS159'),
            (33, 'LIBS162'),
            (34, 'LIBS126'),
            (35, 'corneti LIBS139'), # cornetite
            (36, 'LIBS127'),
            (37, 'LIBS167'),
            (38, 'LIBS185'),
            (39, 'LIBS187'),
            (40, 'LIBS165'),
            (41, 'cupri LIBS044 LIBS045 LIBS046 LIBS047 LIBS048 LIBS049 LIBS051 LIBS198 LIBS199 LIBS200 LIBS201 LIBS202 LIBS203'), # cuprite
            (42, 'LIBS128'),
            (43, 'LIBS145'),
            (44, 'LIBS118'),
            (45, 'LIBS055'),
            (46, 'LIBS097'),
            (47, 'LIBS161'),
            (49, 'LIBS099'),
            (50, 'LIBS138'),
            (52, 'LIBS174'),
            (53, 'LIBS172'),
            (54, 'LIBS136'),
            (55, 'LIBS109'),
            (56, 'LIBS152'),
            (57, 'LIBS188'),
            (58, 'LIBS163'),
            (59, 'LIBS191'),
            (60, 'LIBS122'),
            (61, 'LIBS150'),
            (62, 'LIBS094'),
            (63, 'LIBS151'),
            (64, 'LIBS173'),
            (65, 'LIBS056 LIBS180'),
            (66, 'LIBS116'),
            (67, 'LIBS182'),
            (69, 'LIBS057 LIBS058 LIBS171'),
            (70, 'LIBS189'),
            (71, 'LIBS130'),
            (72, 'LIBS133'),
            (73, 'LIBS059 LIBS060 LIBS061 LIBS062 LIBS063 LIBS064 LIBS065 LIBS066 LIBS067 LIBS068 LIBS069 LIBS070 LIBS071 LIBS072 LIBS073 LIBS074 LIBS075'), # malachite
            (74, 'LIBS110'),
            (75, 'LIBS179'),
            (76, 'LIBS076 LIBS114'),
            (77, 'LIBS193'),
            (78, 'LIBS156'),
            (79, 'LIBS190'),
            (80, 'oliveni LIBS077 LIBS078 LIBS079 LIBS135'), # olivenite
            (81, 'LIBS157'),
            (82, 'LIBS146'),
            (84, 'LIBS177'),
            (85, 'LIBS142'),
            (86, 'pseudomalachi LIBS081 LIBS082 LIBS175'), # pseudomalachite
            (87, 'LIBS132'),
            (88, 'rosasi LIBS192'), # rosasite
            (89, 'LIBS096'),
            (90, 'LIBS084 LIBS134'),
            (91, 'LIBS131'),
            (92, 'LIBS085 LIBS086 LIBS100 LIBS100'),
            (94, 'LIBS149'),
            (95, 'LIBS087'),
            (96, 'LIBS169'),
            (97, 'tenori LIBS155'), # tenorite
            (98, 'tetrahedr LIBS088 LIBS089'), # tetrahedrite
            (99, 'LIBS102'),
            (100, 'LIBS091'),
            (101, 'LIBS184'),
            (102, 'LIBS181'),
            (103, 'LIBS124'),
            (104, 'LIBS092 LIBS147'),
            (105, 'LIBS129'),
            (106, 'LIBS098'),
            (107, 'LIBS093')
        ]

#save the tuples of minerals ID and foldernames
np.save('/Users/jh/github/libs-pewpew/data/mineral_infos/mineral_id_folder_100', minerals_all)

#save only the mineral ids
mineral_all_id = [a_tuple[0] for a_tuple in minerals_all]
np.save('/Users/jh/github/libs-pewpew/data/mineral_infos/mineral_id_100', mineral_all_id)



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

#save the tuples of minerals ID and foldernames
np.save('/Users/jh/github/libs-pewpew/data/mineral_infos/mineral_id_folder_12', minerals_12)

#save only the mineral ids
mineral_12_id = [a_tuple[0] for a_tuple in minerals_12]
np.save('/Users/jh/github/libs-pewpew/data/mineral_infos/mineral_id_12', mineral_12_id)


minerals_12_id_name = [ (26, 'Chalcopyrite'),
                        (98, 'Tetrahedrite'),
                        (19, 'Bornite'),
                        (41, 'Cuprite'),
                        (28, 'Chalcotrichite'),
                        (97, 'Tenorite'),
                        (88, 'Rosasite'),
                        (11, 'Azurit'),
                        (73, 'Malachite'),
                        (86, 'Pseudomalachite'),
                        (80, 'Olivenite'),
                        (35, 'Cornetite')]

#save mineral ID and names for visualisations
np.save('/Users/jh/github/libs-pewpew/data/mineral_infos/mineral_id_name_12', minerals_12_id_name)
