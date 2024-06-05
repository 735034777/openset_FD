import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


# Function to plot and perform linear regression
def plot_metric_vs_openness(df, metric, distance_types, ax):
    for dist_type in distance_types:
        # Filter data for each metric type
        subset = df[df['METRIC_TYPE'] == dist_type]
        # Note: Not setting label here to manage legends globally
        sns.regplot(x='openness', y=metric, data=subset, ax=ax, label='_nolegend_', order=1, scatter_kws={'s': 10},
                    line_kws={'linewidth': 2})

    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    ax.grid(True)


# Function to label subplots with letters inside the top right of the plot area
def label_subplots(axes, labels):
    for ax, label in zip(axes.flatten(), labels):
        ax.text(0.95, 0.95, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')


# Load data from all CSV files in a specified directory
folder_path = 'H:\project\imbalanced_data\openset_FD\data\结果图'
files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
data_frames = [pd.read_csv(file) for file in files]

# Set of metrics and distance types to plot
metrics = ['recall', 'f1', 'youdens_index']
distance_types = ['cosine', 'euclidean', 'manhattan']

# Create labels for subplots based on the number of files and metrics
subplot_labels = ['({})'.format(chr(97 + i)) for i in range(len(data_frames) * len(metrics))]

# Create a large figure to combine all plots into a grid
fig, axes = plt.subplots(nrows=len(data_frames), ncols=3, figsize=(15, 6 * len(data_frames)), sharex=True, sharey='row')

for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        metric = metrics[j]
        df = data_frames[i]
        plot_metric_vs_openness(df, metric, distance_types, ax)
        ax.set_title('')  # Remove titles from subplots

label_subplots(axes, subplot_labels)  # Label subplots with letters inside them
plt.tight_layout()

# Create a single legend for the entire figure
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[:3], labels[:3], loc='upper center', ncol=3, fontsize='large')

plt.show()
