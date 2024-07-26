"""
File: cwru_SNR
Author: admin
Date Created: 2024/6/18
Last Modified: 2024/6/18

Description:
    This file is used to perform dimensionality reduction on the MAV dataset using t-SNE.
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from src.utils.train_base_model import load_data,load_model

class DimensionReducer:
    def __init__(self, data_file_path, label_file_path,pred_label_file_path):
        self.data_file_path = data_file_path
        self.label_file_path = label_file_path
        self.pred_label_file_path = pred_label_file_path
        # self.train_label = load_data("train")

    def read_MAV(self,file_path):
        self.MAVs = np.load(file_path)
        # self.reudced_MAVs = self.tsne.transform(self.MAVs)

    def read_files(self):
        # 读取numpy文件
        self.data = np.load(self.data_file_path)[::10]
        self.labels = np.load(self.label_file_path)[::10]
        self.pred_labels = np.load(self.pred_label_file_path)[::10]


    def reduce_dimension(self, perplexity=300, learning_rate=20):
        # 使用TSNE进行降维
        self.tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate)
        self.reduced_data = self.tsne.fit_transform(self.data)


    def plot_data(self):
        # 创建一个颜色和形状的字典
        markers = ('o', 'p', 's', 'D', 'd',"v","<")
        colors = ('red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown')
        marker_color_dict = {marker: color for marker, color in zip(markers, colors)}

        # 根据标签不同赋予不同的颜色和形状
        # cmap = get_cmap('viridis')
        # for marker, color in marker_color_dict.items():
        for i in range(max(np.unique(self.labels))+1):
            indices = np.where(self.labels == i)
            # a = self.reduced_data[indices, 0]
            plt.scatter(self.reduced_data[indices, 0], self.reduced_data[indices, 1], marker=markers[i], color=colors[i])

        # 添加图例
        custom_lines = [Line2D([0], [0], marker=marker, color=color, markerfacecolor=color, markersize=10) for
                        marker, color in marker_color_dict.items()]
        plt.legend(custom_lines, list(marker_color_dict.keys()))
        plt.show()


class MAVsReducer(DimensionReducer):
    def __init__(self, data_file_path, label_file_path,pred_label_file_path,MAV_file_path):
        super().__init__(data_file_path, label_file_path,pred_label_file_path)
        self.MAV_file_path = MAV_file_path

    def reduce_dimension(self, perplexity=15, learning_rate=20):
        # 使用TSNE进行降维
        self.read_MAV(self.MAV_file_path)
        self.data = np.concatenate([self.MAVs,self.train_data,self.data])
        self.tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate)
        self.reduced_data = self.tsne.fit_transform(self.data)
        self.reduced_MAVs = self.data[:self.MAVs.shape[0]]
        self.reduced_train = self.reduced_data[self.MAVs.shape[0]:self.MAVs.shape[0]+self.train_data.shape[0]]
        self.reduced_data = self.reduced_data[self.MAVs.shape[0]+self.train_data.shape[0]:]

    def read_train(self,path,label_path):
        self.train_data = np.load(path)[::10]
        self.train_label = np.load(label_path)[::10]

    def plot_data(self):
        markers = ('o', 'p', 's', 'D', 'd',"v","<")
        colors = ('red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown')
        marker_color_dict = {marker: color for marker, color in zip(markers, colors)}

        # 根据标签不同赋予不同的颜色和形状
        # cmap = get_cmap('viridis')
        # for marker, color in marker_color_dict.items():
        for i in range(max(np.unique(self.labels))+1):
            indices = np.where(self.labels == i)
            train_index = np.where(self.train_label == i)
            # if indices[0].shape[0]>0 and i<max(np.unique(self.labels)):
            #     plt.scatter(self.reduced_MAVs[i, 0], self.reduced_data[i, 1], marker=markers[i], color="black")
            # a = self.reduced_data[indices, 0]

            plt.scatter(self.reduced_data[indices, 0], self.reduced_data[indices, 1], marker=markers[i], color=colors[i])
            plt.scatter(self.reduced_train[train_index, 0], self.reduced_train[train_index, 1], marker=markers[i], edgecolor=colors[i],facecolor = "none")
            plt.scatter(np.mean(self.reduced_train[train_index, 0]), np.mean(self.reduced_train[train_index, 1]), marker="*", edgecolor=colors[i],facecolor = "none")

        # 添加图例
        labelname =[ "c"+str(i) for i in [1,5,4,6,7,10,8]]

        custom_lines = [Line2D([0], [0], marker=marker, color=color, markerfacecolor=color, markersize=10) for
                        marker, color in marker_color_dict.items()]
        plt.legend(custom_lines, labelname)
        plt.show()






# 使用示例
# reducer = DimensionReducer("activation_vectory_cosine.npy","y_test_cosine.npy","y_predicted_values_cosine.npy")
# # reducer.read_MAV("mean.npy")
# reducer.read_files()
# reducer.reduce_dimension()
# reducer.read_MAV("mean.npy")
# reducer.plot_data()

reducer = MAVsReducer("activation_vectory_cosine.npy","y_test_cosine.npy","y_predicted_values_cosine.npy","mean.npy")
# reducer.read_MAV("mean.npy")
reducer.read_train("SNEtrain_score.npy","train_label.npy")
reducer.read_files()
reducer.reduce_dimension()
# reducer.read_MAV("mean.npy")
reducer.plot_data()


