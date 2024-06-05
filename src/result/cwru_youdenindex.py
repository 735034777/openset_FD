import matplotlib.pyplot as plt
from cwru_recall import colors


def plot_cwru_youden(ax):
    # Organizing data into a list of dictionaries
    datasets = [
        {
            'x': [0.039231, 0.087129, 0.133975, 0.215535, 0.292893],
            'y1': [0.961059, 0.911412, 0.969488, 0.881083, 0.686740],
            'y2': [0.954958, 0.914556, 0.968130, 0.881413, 0.694607],
            'y3': [0.956537, 0.919078, 0.970622, 0.881610, 0.691781]
        },
        {
            'x': [0.039231, 0.074180, 0.122942, 0.215535, 0.292893],
            'y1': [0.982912, 0.874308, 0.868162, 0.815040, 0.737280],
            'y2': [0.980856, 0.901673, 0.868764, 0.808645, 0.692374],
            'y3': [0.982014, 0.903996, 0.860658, 0.813998, 0.682011]
        },
        {
            'x': [0.026671, 0.074180, 0.133975, 0.215535, 0.320634],
            'y1': [0.948056, 0.746109, 0.820692, 0.695922, 0.590243],
            'y2': [0.946597, 0.728655, 0.815612, 0.696229, 0.585331],
            'y3': [0.947983, 0.744346, 0.819356, 0.697479, 0.585587]
        },
        {
            'x': [0.039231, 0.087129, 0.122942, 0.215535, 0.261451],
            'y1': [0.940013, 0.921513, 0.905715, 0.816848, 0.637261],
            'y2': [0.927140, 0.916705, 0.901823, 0.828462, 0.630455],
            'y3': [0.908729, 0.919393, 0.901908, 0.830400, 0.630960]
        }
    ]

    # Colors and markers definition
    colors = [
        'blue', 'green', 'red', 'navy', 'forestgreen', 'crimson',
        'darkviolet', 'teal', 'goldenrod', 'indigo', 'olive', 'maroon'
    ]
    marker = "o"
    labels = [
        'SDOS cosine', 'SDOS euclidean', 'SDOS mahalanobis',
        '0hp->1hp cosine', '0hp->1hp euclidean', '0hp->1hp mahalanobis',
        '0hp->2hp cosine', '0hp->2hp euclidean', '0hp->2hp mahalanobis',
        '1hp->2hp cosine', '1hp->2hp euclidean', '1hp->2hp mahalanobis',
    ]

    # Plotting each dataset with specific colors and markers
    for i, data in enumerate(datasets):
        for j in range(3):  # Assuming there are three y-values per dataset
            # label = f'Dataset {i+1} - Col {j+2}'
            ax.plot(data['x'], data[f'y{j+1}'], color=colors[3*i+j], marker=marker, label=labels[3*i+j])

    # Axis and grid configuration
    # ax.set_ylabel('Youden\'s index')
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7, ncol=1)
    ax.text(0.01, 0.9, '(c)', transform=ax.transAxes, size=15, weight='bold')




if __name__=="__main__":
    # Data preparation for all datasets
    data1 = {
        'x': [0.039231, 0.087129, 0.133975, 0.215535, 0.292893],
        'y1': [0.961059, 0.911412, 0.969488, 0.881083, 0.686740],
        'y2': [0.954958, 0.914556, 0.968130, 0.881413, 0.694607],
        'y3': [0.956537, 0.919078, 0.970622, 0.881610, 0.691781]
    }

    data2 = {
        'x': [0.039231, 0.074180, 0.122942, 0.215535, 0.292893],
        'y1': [0.982912, 0.874308, 0.868162, 0.815040, 0.737280],
        'y2': [0.980856, 0.901673, 0.868764, 0.808645, 0.692374],
        'y3': [0.982014, 0.903996, 0.860658, 0.813998, 0.682011]
    }

    data3 = {
        'x': [0.026671, 0.074180, 0.133975, 0.215535, 0.320634],
        'y1': [0.948056, 0.746109, 0.820692, 0.695922, 0.590243],
        'y2': [0.946597, 0.728655, 0.815612, 0.696229, 0.585331],
        'y3': [0.947983, 0.744346, 0.819356, 0.697479, 0.585587]
    }

    data4 = {
        'x': [0.039231, 0.087129, 0.122942, 0.215535, 0.261451],
        'y1': [0.940013, 0.921513, 0.905715, 0.816848, 0.637261],
        'y2': [0.927140, 0.916705, 0.901823, 0.828462, 0.630455],
        'y3': [0.908729, 0.919393, 0.901908, 0.830400, 0.630960]
    }

    # Plot configuration
    plt.figure(figsize=(14, 8))

    # Plotting each dataset with specific colors and markers
    plt.plot(data1['x'], data1['y1'], 'ro-', label='Dataset 1 - Col 2')
    plt.plot(data1['x'], data1['y2'], 'rx-', label='Dataset 1 - Col 3')
    plt.plot(data1['x'], data1['y3'], 'r*-', label='Dataset 1 - Col 4')

    plt.plot(data2['x'], data2['y1'], 'go-', label='Dataset 2 - Col 2')
    plt.plot(data2['x'], data2['y2'], 'gx-', label='Dataset 2 - Col 3')
    plt.plot(data2['x'], data2['y3'], 'g*-', label='Dataset 2 - Col 4')

    plt.plot(data3['x'], data3['y1'], 'bo-', label='Dataset 3 - Col 2')
    plt.plot(data3['x'], data3['y2'], 'bx-', label='Dataset 3 - Col 3')
    plt.plot(data3['x'], data3['y3'], 'b*-', label='Dataset 3 - Col 4')

    plt.plot(data4['x'], data4['y1'], 'yo-', label='Dataset 4 - Col 2')
    plt.plot(data4['x'], data4['y2'], 'yx-', label='Dataset 4 - Col 3')
    plt.plot(data4['x'], data4['y3'], 'y*-', label='Dataset 4 - Col 4')

    plt.title('Integrated Line Plot of All Data Sets')
    plt.xlabel('X-axis values (Column 1)')
    plt.ylabel('Y-axis values')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
