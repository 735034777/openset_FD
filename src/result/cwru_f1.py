import matplotlib.pyplot as plt
import pandas as pd
from cwru_recall import colors
from matplotlib.ticker import PercentFormatter



if __name__=="__main__":
    # Data definitions
    data1 = {
        'x': [0.039231, 0.087129, 0.133975, 0.215535, 0.292893],
        'y2': [0.968816, 0.918718, 0.979699, 0.854719, 0.625495],
        'y3': [0.963852, 0.923218, 0.978326, 0.854698, 0.634897],
        'y4': [0.965109, 0.925820, 0.979975, 0.854947, 0.631809]
    }
    data2 = {
        'x': [0.039231, 0.074180, 0.122942, 0.215535, 0.292893],
        'y2': [0.985056, 0.907069, 0.862070, 0.724826, 0.629134],
        'y3': [0.980856, 0.901673, 0.868764, 0.808645, 0.692374],
        'y4': [0.984185, 0.932062, 0.855428, 0.734290, 0.560071]
    }
    data3 = {
        'x': [0.026671, 0.074180, 0.133975, 0.215535, 0.320634],
        'y2': [0.953229, 0.787094, 0.721478, 0.662166, 0.505610],
        'y3': [0.952046, 0.768818, 0.704200, 0.663941, 0.522355],
        'y4': [0.953041, 0.783472, 0.714337, 0.664762, 0.525011]
    }
    data4 = {
        'x': [0.039231, 0.087129, 0.122942, 0.215535, 0.261451],
        'y2': [0.947192, 0.942225, 0.890277, 0.782526, 0.524771],
        'y3': [0.935354, 0.939355, 0.884141, 0.793257, 0.511643],
        'y4': [0.918224, 0.941904, 0.883858, 0.795660, 0.512538]
    }

    # Create DataFrames
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df4 = pd.DataFrame(data4)

    # Plotting all datasets
    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'H', 'D', 'X', '+']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'navy', 'teal', 'lime']
    labels = ['Set 1, Col 2', 'Set 1, Col 3', 'Set 1, Col 4',
              'Set 2, Col 2', 'Set 2, Col 3', 'Set 2, Col 4',
              'Set 3, Col 2', 'Set 3, Col 3', 'Set 3, Col 4',
              'Set 4, Col 2', 'Set 4, Col 3', 'Set 4, Col 4']

    # Adding each plot
    for i, df in enumerate([df1, df2, df3, df4]):
        for j in range(3):
            # plt.plot(df['x'], df.iloc[:, j+1], marker=markers[i*3+j], linestyle='-', color=colors[i*3+j], label=labels[i*3+j])
            ax.plot(df['x'], df.iloc[:, j+1], marker=markers[i*3+j], linestyle='-', color=colors[i*3+j], label=labels[i*3+j])

    # Chart formatting
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Integrated Point-Line Plot for All Data Sets')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.grid(True)
    plt.show()

def plot_cwru_f1(ax):
    # Data definitions
    data1 = {
        'x': [0.039231, 0.087129, 0.133975, 0.215535, 0.292893],
        'y2': [0.968816, 0.918718, 0.979699, 0.854719, 0.625495],
        'y3': [0.963852, 0.923218, 0.978326, 0.854698, 0.634897],
        'y4': [0.965109, 0.925820, 0.979975, 0.854947, 0.631809]
    }
    data2 = {
        'x': [0.039231, 0.074180, 0.122942, 0.215535, 0.292893],
        'y2': [0.985056, 0.907069, 0.862070, 0.724826, 0.629134],
        'y3': [0.980856, 0.901673, 0.868764, 0.808645, 0.692374],
        'y4': [0.984185, 0.932062, 0.855428, 0.734290, 0.560071]
    }
    data3 = {
        'x': [0.026671, 0.074180, 0.133975, 0.215535, 0.320634],
        'y2': [0.953229, 0.787094, 0.721478, 0.662166, 0.505610],
        'y3': [0.952046, 0.768818, 0.704200, 0.663941, 0.522355],
        'y4': [0.953041, 0.783472, 0.714337, 0.664762, 0.525011]
    }
    data4 = {
        'x': [0.039231, 0.087129, 0.122942, 0.215535, 0.261451],
        'y2': [0.947192, 0.942225, 0.890277, 0.782526, 0.524771],
        'y3': [0.935354, 0.939355, 0.884141, 0.793257, 0.511643],
        'y4': [0.918224, 0.941904, 0.883858, 0.795660, 0.512538]
    }

    # Create DataFrames
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df4 = pd.DataFrame(data4)

    # Plotting all datasets
    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'H', 'D', 'X', '+']
    markers = ['o' for i in range(12)]
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'navy', 'teal', 'lime']
    colors = [
        'blue', 'green', 'red', 'navy', 'forestgreen', 'crimson',
        'darkviolet', 'teal', 'goldenrod', 'indigo', 'olive', 'maroon'
    ]
    labels = [
        'SDOS cosine', 'SDOS euclidean', 'SDOS mahalanobis',
        '0hp->1hp cosine', '0hp->1hp euclidean', '0hp->1hp mahalanobis',
        '0hp->2hp cosine', '0hp->2hp euclidean', '0hp->2hp mahalanobis',
        '1hp->2hp cosine', '1hp->2hp euclidean', '1hp->2hp mahalanobis',
    ]

    # Adding each plot
    for i, df in enumerate([df1, df2, df3, df4]):
        for j in range(3):
            # plt.plot(df['x'], df.iloc[:, j+1], marker=markers[i*3+j], linestyle='-', color=colors[i*3+j], label=labels[i*3+j])
            ax.plot(df['x'], df.iloc[:, j + 1], marker=markers[i * 3 + j], linestyle='-', color=colors[i * 3 + j],
                    label=labels[i * 3 + j])

    # ax.set_title('Simulated Dataset')
    # ax.set_xlabel('X Axis (0 to 1)')
    # ax.set_ylabel('F1 score')
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(True)
    ax.legend(loc = "upper right",fontsize=7, ncol=1)
    ax.text(0.01, 0.9, '(b)', transform=ax.transAxes, size=15, weight='bold')





