import pickle as pkl
import os

data_set = ['train', 'valid', 'test']
data_type = ['acid', 'iodine']
split_root = '/data/mxj/Cervix/Segmentation/cervix_resize_600_segmentation/data_split/'

data_label = pkl.load(open('/data/mxj/Cervix/Segmentation/cervix_resize_600_segmentation/data_split/'
                           'Classification/label_from_seg_annotation.pkl', 'rb'))
save_root = '/data/lxc/Cervix/pos_only'

for type in data_type:
    for set in data_set:
        split_path = os.path.join(split_root, type, "{}.txt".format(set))
        save_path = os.path.join(save_root, type)
        save_file = os.path.join(save_path, "{}_pos.txt".format(set))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        result = []
        with open(split_path, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                line = line.strip()
                name = line[:-2]
                type_num = line[-2:]
                if name in data_label.keys() and data_label[name] != 'lsil_n':
                    result.append(line)

            with open(save_file, 'w') as f:
                for line in result:
                    f.write(line + '\n')
                print('file save at: ', save_file)