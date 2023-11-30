import scipy.io as scio
import re
import numpy as np
import os


raw_num = 240
col_num = 2000
class Data_CWRU(object):

    def __init__(self):
        self.data = self.get_data()
        self.label = self.get_label()

    def file_list(self):
        return os.listdir('data/')

    def get_data(self):
        file_list = self.file_list()
        for i in range(len(file_list)):
            file = scio.loadmat('data/{}'.format(file_list[i]))
            for k in file.keys():
                file_matched = re.match('X\d{3}_DE_time', k)
                if file_matched:
                    key = file_matched.group()
            if i == 0:
                data = np.array(file[key][0:480000].reshape(raw_num, col_num))
            else:
                data = np.vstack((data, file[key][0:480000].reshape((raw_num, col_num))))
        return data

    def get_label(self):
        file_list = self.file_list()
        title = np.array([i.replace('.mat', '') for i in file_list])
        label = title[:, np.newaxis]
        label_copy = np.copy(label)
        for _ in range(raw_num - 1):
            label = np.hstack((label, label_copy))
        return label.flatten()