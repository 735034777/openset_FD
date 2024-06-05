import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_metrics_with_fits(data, metric_names, distance_metrics):
    plt.figure(figsize=(18, 6))

    # Loop through each metric to plot
    for i, metric in enumerate(metric_names):
        plt.subplot(1, len(metric_names), i + 1)
        for distance in distance_metrics:
            filtered_data = data[data['METRIC_TYPE'] == distance]
            sns.scatterplot(x='openness', y=metric, data=filtered_data, label=f'{distance}', s=50)

            # Linear fit
            coefficients = np.polyfit(filtered_data['openness'], filtered_data[metric], 1)
            poly = np.poly1d(coefficients)
            plt.plot(filtered_data['openness'], poly(filtered_data['openness']),
                     label=f'Fit: {coefficients[0]:.2f}x + {coefficients[1]:.2f}')

        plt.title(f'{metric} by Openness')
        plt.xlabel('Openness')
        plt.ylabel(metric)
        plt.legend(title='Metric Type')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_metrics_with_fits(data, ['recall', 'f1', 'youdens_index'], ['cosine', 'euclidean', 'manhattan'])
if __name__ =="__main__":
    file_path = r"H:\project\imbalanced_data\openset_FD\src\train\CWRU\result\modify_CDOS_0_2_max_youdens.csv"
    # data = pd.read_csv(file_path)
    # metric_names = ['recall', 'f1', 'youdens_index']
    # distance_metrics = ['cosine', 'euclidean', 'manhattan']
    # plot_metrics_with_fits(data, metric_names, distance_metrics)
    # Python script to read the first few lines of a CSV file to inspect the headers and data

    # file_path = 'your_file_path.csv'  # Replace 'your_file_path.csv' with the actual path of your CSV file

    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read the first line for headers
            headers = file.readline()
            print("Headers:", headers.strip())  # Print headers without additional newline

            # Read the next few lines to inspect the data
            for _ in range(5):  # Adjust the range to read more lines if necessary
                data_line = file.readline()
                print(data_line.strip())  # Print each data line without additional newline

    except Exception as e:
        print("Error reading file:", e)
