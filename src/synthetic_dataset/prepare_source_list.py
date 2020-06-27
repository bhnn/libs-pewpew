import csv
import numpy as np

if __name__ == '__main__':
    with open('data/synthetic_minerals_raw.csv', newline='') as file:
        f = csv.reader(file)
        
        # create lists of class and group names
        classes = list()
        groups = list()
        for _, name, _, _, m_class, m_group in f:
            classes.append(m_class)
            groups.append(m_group if m_group else f'zzz_placeholder_{name}')
        
        # lower-case and strip whitespace from classes, create alphabetic indexing of names
        cls_dict = dict()
        classes = sorted(classes)
        class_index = 0
        for cls in classes:
            cls = cls.lower().strip()
            if cls not in cls_dict:
                cls_dict[cls] = class_index
                class_index += 1
        
        # lower-case and strip whitespace from groups, create alphabetic indexing, create placeholder index for missing groups
        grp_dict = dict()
        groups = sorted(groups)
        group_index = 0
        for grp in groups:
            grp = grp.lower().strip()
            if grp not in grp_dict:
                grp_dict[grp] = group_index
                group_index += 1

        # print list of collected classes and groups for visual inspection
        print('classes: ')
        for k,v in cls_dict.items():
            print(k,v)

        print('\nsubgroups:')
        for k,v in grp_dict.items():
            print(k,v)
        print('')

    with open('data/synthetic_minerals_raw.csv', newline='') as file:
        # traversing the reader consumes the input, needs to be reopened
        f = csv.reader(file)

        ids = list()
        names = list()
        final = list()
        for num, name, elements, composition, m_class, m_group in f:
            # pre-clean class and group names
            num = int(num)
            m_class = m_class.lower().strip()
            m_group = m_group.lower().strip()

            # verify indexing integrity
            if ids:
                assert num - ids[-1] == 1, f'indexing gap found for \'{name}\' ({num})'
            
            assert num not in ids, f'duplicate ids found for \'{name}\' ({num})'
            ids.append(num)

            # check for duplicate elements
            assert name not in names, f'duplicate element names found for \'{name}\' ({num})'
            names.append(name)

            # verify amount of elements matches amount of compositions
            elements = elements.strip('[]').replace(' ', '').split(',')
            composition = np.array([float(i) for i in composition.strip('[]').replace(']','').split(',')])
            assert len(elements) == len(composition), f'mismatch in amount of elements/compositions found for \'{name}\' ({num})'

            # verify sum of composition is valid
            assert 99.9 < sum(composition) < 100.999, f'invalid sum of atomic composition for \'{name}\' ({num})'

            # verify mineral class
            assert m_class, f'Missing mineral class found for \'{name}\' ({num})'
            
            # index of original class if present, index of placeholder if element has no group
            grp = grp_dict[m_group] if m_group else grp_dict[f'zzz_placeholder_{name}']
            final.append((num, name, elements, composition, cls_dict[m_class], grp))
        npy = np.array(final)
        np.save('data/synthetic_minerals', npy)