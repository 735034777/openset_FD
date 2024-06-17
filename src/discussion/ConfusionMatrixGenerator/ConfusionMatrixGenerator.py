import numpy as np
import numpy as np
import matplotlib.pyplot as plt

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def set_confusion_matrix(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update_confusion_matrix(self, predictions, true_labels):
        remapped_predictions = self.remap_labels(predictions)
        remapped_true_labels = self.remap_labels(true_labels)
        num = 0

        for i in range(len(predictions)):
            self.confusion_matrix[remapped_predictions[i]][remapped_true_labels[i]] += 1
            if remapped_predictions[i]!=remapped_true_labels[i]:
                num = num+1
        print(num)
    def get_confusion_matrix(self):
        return self.confusion_matrix

    def remap_labels(self, labels):
        unique_labels = np.unique(labels)
        label_map = {label: index for index, label in enumerate(unique_labels)}
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
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        # Label the axes
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        # Add the number of predictions for each cell
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(j, i, self.confusion_matrix[i, j], ha='center', va='center', color='w')

        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        # Label the axes
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

    def plot_confusion_matrix(self, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        # If normalize is True, the confusion matrix is normalized
        if normalize:
            self.confusion_matrix = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:,
                                                                            np.newaxis]

        # Plot the confusion matrix
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        # Label the axes
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        # Add the number of predictions for each cell
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(j, i, self.confusion_matrix[i, j], ha='center', va='center', color='w')

        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        # Label the axes
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

    # This function reads the predictions and true labels from the files and updates the conf# This function reads the predictions and true labels from the files and updates the confusion matrix
    def read_from_files(self, pred_file_path, true_file_path):
        # Load the predictions from the file
        pred_y = np.load(pred_file_path)
        # Load the true labels from the file
        true_y = np.load(true_file_path)
        self.labels = np.unique(true_y)
        self.num_classes = self.labels.shape[0]
        self.set_confusion_matrix()
        # Update the confusion matrix with the predictions and true labels
        self.update_confusion_matrix(pred_y, true_y)




if __name__ == '__main__':
    # Example usage:
    confusion_matrix = ConfusionMatrix(7)
    confusion_matrix.read_from_files('y_predicted_values_cosine.npy', 'y_test_cosine.npy')
    confusion_matrix.plot_confusion_matrix()
    plt.show()

    # Example usage:
    confusion_matrix = ConfusionMatrix(3)
    confusion_matrix.read_from_files('y_softmax_predicted_values_cosine.npy', 'y_test_cosine.npy')
    confusion_matrix.plot_confusion_matrix()
    plt.show()

    # # Example usage:
    # confusion_matrix = ConfusionMatrix(3)
    # confusion_matrix.update_confusion_matrix([0, 1, 2], [0, 1, 2])
    # print(confusion_matrix.get_confusion_matrix())