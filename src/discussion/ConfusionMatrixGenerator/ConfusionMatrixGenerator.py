import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import torch

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def set_confusion_matrix(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update_confusion_matrix(self, predictions, true_labels):
        # remapped_true_labels = self.remap_labels(true_labels)
        # remapped_predictions = self.remap_labels(predictions,self.label_map)
        for i in range(len(predictions)):
            # self.confusion_matrix[predictions[i]][true_labels[i]] += 1
            self.confusion_matrix[true_labels[i]][predictions[i]] += 1

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def remap_labels(self, labels,maps=None):
        if maps is None:
            unique_labels = np.unique(labels)
            label_map = {label: index for index, label in enumerate(unique_labels)}
        else:
            label_map = maps

        self.label_map = label_map  # 将映射关系作为类的属性
        remapped_labels = [label_map[label] for label in labels]

        return remapped_labels



    def get_confusion_matrix(self):
        return self.confusion_matrix

    def plot_confusion_matrix(self, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        # If normalize is True, the confusion matrix is normalized
        if normalize:
            self.confusion_matrix = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:,
                                                                            np.newaxis]

        # Plot the confusion matrix
        plt.figure(figsize=(9, 9))
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap)
        # plt.title(title)
        # plt.colorbar()
        # Label the axes
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        # Add the number of predictions for each cell
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(j, i, int(self.confusion_matrix[i, j]), ha='center', va='center',
                         color='white' if self.confusion_matrix[i,j]>100 else 'black',fontsize = 20)

        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap)
        # plt.title(title)
        # plt.colorbar()
        # Label the axes
        labelname = ["c" + str(i) for i in [1, 5, 4, 6, 7,10,8]]
        tick_marks = np.arange(self.num_classes)
        # tick_marks =labelname
        # 使用plt.xticks设置横坐标轴的标签
        plt.xticks(range(len(labelname)), labelname, rotation=45,fontsize = 22)

        # 使用plt.yticks设置纵坐标轴的标签
        plt.yticks(range(len(labelname)), labelname,fontsize = 22)
        plt.xlabel('Predicted label',fontsize = 22)
        plt.ylabel('True label',fontsize = 22)
        plt.savefig("./confusion_matrix_"+metric+".png")

    # def plot_confusion_matrix(self, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    #     # If normalize is True, the confusion matrix is normalized
    #     if normalize:
    #         self.confusion_matrix = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:,
    #                                                                         np.newaxis]
    #
    #     # Plot the confusion matrix
    #     plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()
    #     # Label the axes
    #     tick_marks = np.arange(self.num_classes)
    #     plt.xticks(tick_marks, tick_marks, rotation=45)
    #     plt.yticks(tick_marks, tick_marks)
    #     plt.xlabel('Predicted label')
    #     plt.ylabel('True label')
    #
    #     # Add the number of predictions for each cell
    #     for i in range(self.num_classes):
    #         for j in range(self.num_classes):
    #             plt.text(j, i, int(self.confusion_matrix[i, j]), ha='center', va='center', color='r')
    #
    #     plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()
    #     # Label the axes
    #     tick_marks = np.arange(self.num_classes)
    #     plt.xticks(tick_marks, tick_marks, rotation=45)
    #     plt.yticks(tick_marks, tick_marks)
    #     plt.xlabel('Predicted label')
    #     plt.ylabel('True label')

    # This function reads the predictions and true labels from the files and updates the conf# This function reads the predictions and true labels from the files and updates the confusion matrix
    def read_from_files(self, pred_file_path, true_file_path):
        # Load the predictions from the file
        pred_y = np.load(pred_file_path)
        # Load the true labels from the file
        true_y = np.load(true_file_path)
        self.labels = np.unique(true_y)
        self.num_classes = max(self.labels)+1
        self.set_confusion_matrix()
        # Update the confusion matrix with the predictions and true labels
        self.update_confusion_matrix(pred_y, true_y)

    def read_from_av_file(self,av_filepath,true_file_path):
        av = np.load(av_filepath)
        pred_y = np.argmax(np.array(softmax(torch.tensor(av))),axis=1)
        true_y = np.load(true_file_path)
        self.labels = np.unique(true_y)
        self.num_classes = max(self.labels)+1
        self.set_confusion_matrix()
        # Update the confusion matrix with the predictions and true labels
        self.update_confusion_matrix(pred_y, true_y)




if __name__ == '__main__':
    # Example usage:
    metrics=["cosine","euclidean","manhattan"]
    for metric in metrics:
        confusion_matrix = ConfusionMatrix(7)
        confusion_matrix.read_from_files('y_predicted_values_'+metric+'.npy', 'y_test_'+metric+'.npy')
        # confusion_matrix.read_from_av_file("activation_vectory_"+metric+".npy",'y_test_'+metric+'.npy')
        confusion_matrix.plot_confusion_matrix()
        # plt.show()

    # Example usage:
    # for metric in metrics:
    metric = "softmax"
    confusion_matrix = ConfusionMatrix(3)
    confusion_matrix.read_from_files('y_softmax_predicted_values_cosine.npy', 'y_test_cosine.npy')
    confusion_matrix.plot_confusion_matrix()
    # plt.show()

    # # Example usage:
    # confusion_matrix = ConfusionMatrix(3)
    # confusion_matrix.update_confusion_matrix([0, 1, 2], [0, 1, 2])
    # print(confusion_matrix.get_confusion_matrix())