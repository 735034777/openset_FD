import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# colors = [[246,111,105],[254,179,174],[21,151,165],[14,96,107],[144,201,231],[2,48,74],[]]
# colors = [
#     [1, 182/255, 193/255],   # 亮粉红色
#     [230/255, 190/255, 1],   # 淡紫色
#     [173/255, 216/255, 230/255],  # 淡蓝色
#     [1, 250/255, 205/255],   # 柠檬黄
#     [189/255, 252/255, 201/255],  # 薄荷绿
#     [1, 165/255, 0],         # 桔橙色
#     [1, 223/255, 186/255],   # 淡金色
#     [176/255, 196/255, 222/255],  # 灰蓝色
#     [238/255, 130/255, 238/255],  # 柔和的紫罗兰
#     [220/255, 220/255, 220/255],  # 浅灰色
#     [70/255, 130/255, 180/255],   # 海蓝色
#     [210/255, 180/255, 140/255]   # 浅棕色
# ]

colors = [
    'blue', 'green', 'red', 'navy', 'forestgreen', 'crimson',
    'darkviolet', 'teal', 'goldenrod', 'indigo', 'olive', 'maroon'
]
if __name__=="_main__":
    # Data arrays for x values
    x_datasets = [
        np.array([0.039231, 0.087129, 0.122942, 0.215535, 0.261451]),  # Dataset 1
        np.array([0.039231, 0.07418, 0.122942, 0.215535, 0.320634]),   # Dataset 2
        np.array([0.026671, 0.07418, 0.183975, 0.215535, 0.320634]),    # Dataset 3
        np.array([0.039231, 0.087129, 0.122942, 0.215535, 0.261451])    # Dataset 4
    ]

    # Data arrays for y values
    y_datasets = [
        np.array([0.947469, 0.930694, 0.928505, 0.877178, 0.757505]),  # Dataset 1 Y1
        np.array([0.936221, 0.926305, 0.92558, 0.884938, 0.752864]),   # Dataset 1 Y2
        np.array([0.920057, 0.928636, 0.925691, 0.885952, 0.753201]),  # Dataset 1 Y3
        np.array([0.952767, 0.774956, 0.867512, 0.782889, 0.733501]),  # Dataset 2 Y1
        np.array([0.983295, 0.91783, 0.894101, 0.865073, 0.799188]),    # Dataset 2 Y2
        np.array([0.98431, 0.919483, 0.888315, 0.869119, 0.792161]),    # Dataset 2 Y3
        np.array([0.952767, 0.774956, 0.867512, 0.782889, 0.733501]),  # Dataset 3 Y1
        np.array([0.951454, 0.760082, 0.864269, 0.782956, 0.727075]),  # Dataset 3 Y2
        np.array([0.952715, 0.773689, 0.866873, 0.784033, 0.727015]),  # Dataset 3 Y3
        np.array([0.947469, 0.930694, 0.928505, 0.877178, 0.757505]),  # Dataset 4 Y1
        np.array([0.936221, 0.926305, 0.92558, 0.884938, 0.752864]),   # Dataset 4 Y2
        np.array([0.920057, 0.928636, 0.925691, 0.885952, 0.753201])   # Dataset 4 Y3
    ]

    # Define colors for clarity in the plot
    colors = [
        [1, 182 / 255, 193 / 255],  # 亮粉红色
        [230 / 255, 190 / 255, 1],  # 淡紫色
        [173 / 255, 216 / 255, 230 / 255],  # 淡蓝色
        [1, 250 / 255, 205 / 255],  # 柠檬黄
        [189 / 255, 252 / 255, 201 / 255],  # 薄荷绿
        [1, 165 / 255, 0],  # 桔橙色
        [1, 223 / 255, 186 / 255],  # 淡金色
        [176 / 255, 196 / 255, 222 / 255],  # 灰蓝色
        [238 / 255, 130 / 255, 238 / 255],  # 柔和的紫罗兰
        [220 / 255, 220 / 255, 220 / 255],  # 浅灰色
        [70 / 255, 130 / 255, 180 / 255],  # 海蓝色
        [210 / 255, 180 / 255, 140 / 255]  # 浅棕色
    ]


    # Labels for each line
    labels = [
        'Dataset 1 Y1', 'Dataset 1 Y2', 'Dataset 1 Y3',
        'Dataset 2 Y1', 'Dataset 2 Y2', 'Dataset 2 Y3',
        'Dataset 3 Y1', 'Dataset 3 Y2', 'Dataset 3 Y3',
        'Dataset 4 Y1', 'Dataset 4 Y2', 'Dataset 4 Y3'
    ]

    # Create the combined plot
    plt.figure(figsize=(12, 8))
    for i in range(12):
        plt.plot(x_datasets[i % 4], y_datasets[i], label=labels[i], color=colors[i], marker='o')

    plt.title('Combined Line Plot of All Datasets')
    plt.xlabel('X Axis')
    plt.ylabel('Y Values')
    plt.legend()
    plt.grid(True)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


def plot_cwru_recall(ax):
    # Data definitions
    data1 = {
        "X": [0.039231077, 0.087129071, 0.133974596, 0.215535459, 0.292893219],
        "Y1": [0.968802698, 0.928477040, 0.974337769, 0.922536804, 0.802843551],
        "Y2": [0.963957448, 0.930708865, 0.973275066, 0.922672948, 0.807339327],
        "Y3": [0.965222203, 0.934644120, 0.975378511, 0.922877161, 0.805726159]
    }
    data2 = {
        "X": [0.039231077, 0.0741799, 0.122941981, 0.215535459, 0.292893219],
        "Y1": [0.985063870, 0.899588614, 0.894205419, 0.871411900, 0.827046048],
        "Y2": [0.983295439, 0.917829768, 0.894100987, 0.865072726, 0.799187655],
        "Y3": [0.984309522, 0.919483183, 0.888314587, 0.869118698, 0.792161235]
    }
    data3 = {
        "X": [0.026671473, 0.0741799, 0.133974596, 0.215535459, 0.32063378],
        "Y1": [0.952766763, 0.774956355, 0.867512201, 0.782889020, 0.733500579],
        "Y2": [0.951454026, 0.760082361, 0.864268997, 0.782956400, 0.727074548],
        "Y3": [0.952714525, 0.773688889, 0.866873059, 0.784032865, 0.727014510]
    }
    data4 = {
        "X": [0.039231077, 0.087129071, 0.122941981, 0.215535459, 0.261451054],
        "Y1": [0.947469106, 0.930694203, 0.928504666, 0.877177810, 0.757504888],
        "Y2": [0.936221439, 0.926304727, 0.925580032, 0.884937594, 0.752863886],
        "Y3": [0.920056815, 0.928635706, 0.925691194, 0.885952296, 0.753201268]
    }

    styles = ['o-' for i in range(12)]
    colors = ['blue', 'green', 'red', 'navy', 'forestgreen', 'crimson',
              'darkviolet', 'teal', 'goldenrod', 'indigo', 'olive', 'maroon']
    labels = [
        'SDOS cosine', 'SDOS euclidean', 'SDOS mahalanobis',
        '0hp->1hp cosine', '0hp->1hp euclidean', '0hp->1hp mahalanobis',
        '0hp->2hp cosine', '0hp->2hp euclidean', '0hp->2hp mahalanobis',
        '1hp->2hp cosine', '1hp->2hp euclidean', '1hp->2hp mahalanobis',
    ]

    datasets = [data1, data2, data3, data4]

    for data_dict, i in zip(datasets, range(0, len(styles), 3)):
        data = pd.DataFrame(data_dict)
        for j in range(3):
            ax.plot(data["X"], data[f"Y{j + 1}"], styles[i + j], color=colors[i + j], label=labels[i + j])
    
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    # ax.set_ylabel('Recall')
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7, ncol=1)
    ax.text(0.01, 0.9, '(a)', transform=ax.transAxes, size=15, weight='bold')
