import matplotlib.pyplot as plt
import pandas as pd


import pandas as pd
import matplotlib.pyplot as plt


def plot_pctran_f1(ax):
    # Data as a list of dictionaries
    data_list = [
{
        "X": [0.039231077, 0.133974596, 0.2441, 0.3406, 0.4343],
        "Y1": [0.985207126, 0.8963, 0.8045, 0.7153, 0.6008],
        "Y2": [0.982928923, 0.8973, 0.7866, 0.7182, 0.6635],
        "Y3": [0.973791462, 0.9003, 0.7716, 0.7327, 0.6582]
    },

 {
        "X": [0.039231077, 0.133974596, 0.233035011, 0.367544468, 0.422649731],
        "Y1": [0.9651, 0.942428787, 0.816428199, 0.693020651, 0.559052407],
        "Y2": [0.9579, 0.945507674, 0.814650548, 0.679599882, 0.564084984],
        "Y3": [0.9604, 0.94989348, 0.809003541, 0.68479483, 0.562810609]
    },

 {
        "X": [0.039231077, 0.147197135, 0.233035011, 0.3827866, 0.445299804],
        "Y1": [0.9828, 0.987773857, 0.857873823, 0.71685453, 0.618113118],
        "Y2": [0.9820, 0.978845252, 0.859094082, 0.724284525, 0.636087123],
        "Y3": [0.9750, 0.961660319, 0.856785847, 0.725591059, 0.635070849]
    },

    {
        "X": [0.039231077, 0.133974596, 0.257218647, 0.3827866, 0.445299804],
        "Y1": [0.9595, 0.918159904, 0.791491084, 0.699414559, 0.605748694],
        "Y2": [0.9587, 0.910718551, 0.793528333, 0.678852243, 0.60713738],
        "Y3": [0.9585, 0.919167185, 0.791183966, 0.666078353, 0.607819595]
    }
    ]

    # Markers and line styles
    markers = ['o' for i in range(12)]  # Extend or modify as needed
    line_styles = ['-' for i in range(12)]  # Extend or modify as needed

    # Labels for different datasets
    labels = [
        ['SDOS cosine', 'SDOS euclidean', 'SDOS mahalanobis'],
        ['50%->100% cosine', '50%->100% euclidean', '50%->100% mahalanobis'],
        ['50%->70% cosine', '50%->70% euclidean', '50%->70% mahalanobis'],
        ['70%->100% cosine', '70%->100% euclidean', '70%->100% mahalanobis']
    ]

    # List of 12 colors
    colors = [
        'blue', 'green', 'red', 'navy', 'forestgreen', 'crimson',
        'darkviolet', 'teal', 'goldenrod', 'indigo', 'olive', 'maroon'
    ]
    color_cycle = (color for color in colors)  # Create a single generator for colors

    # Loop through datasets
    for idx, data in enumerate(data_list):
        df = pd.DataFrame(data)
        marker = markers[idx % len(markers)]
        line_style = line_styles[idx % len(line_styles)]
        current_labels = labels[idx]

        for i, y_col in enumerate(['Y1', 'Y2', 'Y3']):
            # Use next(color_cycle) to fetch the next color
            ax.plot(df["X"], df[y_col], marker=marker, linestyle=line_style, color=next(color_cycle),
                    label=current_labels[i])

    # ax.set_ylabel('F1 score')
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7, ncol=1)
    ax.text(0.01, 0.9, '(e)', transform=ax.transAxes, size=15, weight='bold')


if __name__=="__main__":
    # Datasets
    data1 = {
        "X": [0.039231077, 0.133974596, 0.2441, 0.3406, 0.4343],
        "Y1": [0.985207126, 0.8963, 0.8045, 0.7153, 0.6008],
        "Y2": [0.982928923, 0.8973, 0.7866, 0.7182, 0.6635],
        "Y3": [0.973791462, 0.9003, 0.7716, 0.7327, 0.6582]
    }

    data2 = {
        "X": [0.039231077, 0.133974596, 0.233035011, 0.367544468, 0.422649731],
        "Y1": [0.9651, 0.942428787, 0.816428199, 0.693020651, 0.559052407],
        "Y2": [0.9579, 0.945507674, 0.814650548, 0.679599882, 0.564084984],
        "Y3": [0.9604, 0.94989348, 0.809003541, 0.68479483, 0.562810609]
    }

    data3 = {
        "X": [0.039231077, 0.147197135, 0.233035011, 0.3827866, 0.445299804],
        "Y1": [0.9828, 0.987773857, 0.857873823, 0.71685453, 0.618113118],
        "Y2": [0.9820, 0.978845252, 0.859094082, 0.724284525, 0.636087123],
        "Y3": [0.9750, 0.961660319, 0.856785847, 0.725591059, 0.635070849]
    }

    data4 = {
        "X": [0.039231077, 0.133974596, 0.257218647, 0.3827866, 0.445299804],
        "Y1": [0.9595, 0.918159904, 0.791491084, 0.699414559, 0.605748694],
        "Y2": [0.9587, 0.910718551, 0.793528333, 0.678852243, 0.60713738],
        "Y3": [0.9585, 0.919167185, 0.791183966, 0.666078353, 0.607819595]
    }

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df4 = pd.DataFrame(data4)

    # Plotting
    plt.figure(figsize=(14, 8))

    # Data 1
    plt.plot(df1["X"], df1["Y1"], marker='o', linestyle='-', color='b', label='Y1 (Data 1)')
    plt.plot(df1["X"], df1["Y2"], marker='o', linestyle='-', color='r', label='Y2 (Data 1)')
    plt.plot(df1["X"], df1["Y3"], marker='o', linestyle='-', color='g', label='Y3 (Data 1)')

    # Data 2
    plt.plot(df2["X"], df2["Y1"], marker='s', linestyle='--', color='b', label='Y1 (Data 2)')
    plt.plot(df2["X"], df2["Y2"], marker='s', linestyle='--', color='r', label='Y2 (Data 2)')
    plt.plot(df2["X"], df2["Y3"], marker='s', linestyle='--', color='g', label='Y3 (Data 2)')

    # Data 3
    plt.plot(df3["X"], df3["Y1"], marker='^', linestyle='-.', color='b', label='Y1 (Data 3)')
    plt.plot(df3["X"], df3["Y2"], marker='^', linestyle='-.', color='r', label='Y2 (Data 3)')
    plt.plot(df3["X"], df3["Y3"], marker='^', linestyle='-.', color='g', label='Y3 (Data 3)')

    # Data 4
    plt.plot(df4["X"], df4["Y1"], marker='x', linestyle=':', color='b', label='Y1 (Data 4)')
    plt.plot(df4["X"], df4["Y2"], marker='x', linestyle=':', color='r', label='Y2 (Data 4)')
    plt.plot(df4["X"], df4["Y3"], marker='x', linestyle=':', color='g', label='Y3 (Data 4)')

    # Setup plot details
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Integrated Line Plot of All Datasets')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()
