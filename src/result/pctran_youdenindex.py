import matplotlib.pyplot as plt
import pandas as pd


def plot_pctran_youdenindex(ax):
    """
    Plot data grouped by every 5 rows with predefined labels, markers, and colors.

    Parameters:
    - data: DataFrame with columns 'X', 'Y1', 'Y2', 'Y3'
    """
    data = {
        "X": [
            0.039231077, 0.133974596, 0.244071054, 0.340619527, 0.434314575,  # Set 1
            0.054094697, 0.133974596, 0.233035011, 0.367544468, 0.422649731,  # Set 2
            0.036375888, 0.147197135, 0.233035011, 0.3827866, 0.445299804,  # Set 3
            0.014389239, 0.133974596, 0.257218647, 0.3827866, 0.445299804  # Set 4
        ],
        "Y1": [
            0.9793, 0.9305, 0.8741, 0.8660, 0.8298,  # Set 1
            0.9297, 0.9433, 0.8873, 0.8466, 0.7906,  # Set 2
            0.9521, 0.9768, 0.9206, 0.8785, 0.7834,  # Set 3
            0.8647, 0.9196, 0.8922, 0.8717, 0.7956  # Set 4
        ],
        "Y2": [
            0.9759, 0.9257, 0.8612, 0.8668, 0.8764,  # Set 1
            0.9278, 0.9500, 0.8866, 0.8197, 0.7947,  # Set 2
            0.9384, 0.9688, 0.9193, 0.8755, 0.8100,  # Set 3
            0.8527, 0.9050, 0.8977, 0.8533, 0.7956  # Set 4
        ],
        "Y3": [
            0.9606, 0.9349, 0.8476, 0.8805, 0.8623,  # Set 1
            0.9213, 0.9570, 0.8784, 0.8300, 0.7909,  # Set 2
            0.9452, 0.9528, 0.9101, 0.8766, 0.7961,  # Set 3
            0.8716, 0.9166, 0.8968, 0.8392, 0.7961  # Set 4
        ]
    }
    # Predefined labels, markers, and colors for up to 12 groups
    markers = ['o' for i in range(12)]
    colors = ['blue', 'green', 'red', 'navy', 'forestgreen', 'crimson',
              'darkviolet', 'teal', 'goldenrod', 'indigo', 'olive', 'maroon']
    labels = [
        'SDOS cosine', 'SDOS euclidean', 'SDOS mahalanobis',
        '50%->100% cosine', '50%->100% euclidean', '50%->100% mahalanobis',
        '50%->70% cosine', '50%->70% euclidean', '50%->70% mahalanobis',
        '70%->100% cosine', '70%->100% euclidean', '70%->100% mahalanobis',
    ]

    df = pd.DataFrame(data)
    grouped_data = {f'Group {i // 5 + 1}': df.iloc[i:i + 5] for i in range(0, len(df), 5)}

    # plt.figure(figsize=(12, 8))
    for idx, (key, group_data) in enumerate(grouped_data.items()):
        color_idx = idx % len(colors)  # Cycle through colors if fewer than 12 groups
        # ax.scatter(group_data['X'], group_data['Y1'], color=colors[color_idx], label=f'{labels[idx]} - Y1',
        #             marker=markers[idx])
        # ax.scatter(group_data['X'], group_data['Y2'], color=colors[color_idx], label=f'{labels[idx]} - Y2',
        #             marker=markers[idx])
        # ax.scatter(group_data['X'], group_data['Y3'], color=colors[color_idx], label=f'{labels[idx]} - Y3',
        #             marker=markers[idx])
        ax.plot(group_data['X'], group_data['Y1'], "o",color=colors[color_idx*3], linestyle='-',label=labels[idx*3])
        ax.plot(group_data['X'], group_data['Y2'], "o",color=colors[color_idx*3+1], linestyle='-',label=labels[idx*3+1])
        ax.plot(group_data['X'], group_data['Y3'], "o",color=colors[color_idx*3+2], linestyle='-',label=labels[idx*3+2])

    # ax.title('Scatter plot with Line Graph by Groups')
    # ax.set_ylabel('Youden\'s index')
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7, ncol=1)
    ax.text(0.01, 0.9, '(f)', transform=ax.transAxes, size=15, weight='bold')


# def plot_pctran_youdenindex(ax):
#     # Data for plotting
#     data_list = [
#         {
#             "X": [0.039231077, 0.07001889, 0.133974596, 0.209430585, 0.367544468],
#             "Y1": [0.979304076, 0.919344924, 0.886044521, 0.855831431, 0.807674437],
#             "Y2": [0.975879418, 0.912772216, 0.775466134, 0.862911774, 0.807129696],
#             "Y3": [0.960616438, 0.912460485, 0.879204718, 0.869203903, 0.807402067]
#         },
#         {
#             "X": [0.054094697, 0.133974596, 0.233035011, 0.367544468, 0.422649731],
#             "Y1": [0.929745597, 0.943333822, 0.887335756, 0.846575342, 0.790555155],
#             "Y2": [0.92778865, 0.949954216, 0.886607419, 0.819726027, 0.794655732],
#             "Y3": [0.921318493, 0.957032452, 0.878446115, 0.83, 0.790915645]
#         },
#         # Add more datasets here...
#         {
#             "X": [0.036375888, 0.147197135, 0.233035011, 0.3827866, 0.445299804],
#             "Y1": [0.952054795, 0.976788432, 0.920648, 0.878543203, 0.783442053],
#             "Y2": [0.938356164, 0.968797565, 0.919336011, 0.875500527, 0.810013046],
#             "Y3": [0.945205479, 0.95281583, 0.910105499, 0.87664647, 0.807751685]
#         },
#         {
#             "X": [0.014389239, 0.133974596, 0.257218647, 0.3827866, 0.445299804],
#             "Y1": [0.864726027, 0.91957732, 0.892220914, 0.871667545, 0.795596869],
#             "Y2": [0.852739726, 0.904966669, 0.897700366, 0.853345627, 0.795564253],
#             "Y3": [0.871575342, 0.916612336, 0.896770594, 0.839225501, 0.796129593]
#         }
#     ]
#
#     # Labels for each Y series in the datasets
#     labels = [
#         ["Data 1 - Y1", "Data 1 - Y2", "Data 1 - Y3"],
#         ["Data 2 - Y1", "Data 2 - Y2", "Data 2 - Y3"],
#         # Add more label sets here...
#         ["Data 1 - Y1", "Data 1 - Y2", "Data 1 - Y3"],
#         ["Data 2 - Y1", "Data 2 - Y2", "Data 2 - Y3"],
#     ]
#
#     # Colors for each dataset
#     colors = [
#         'blue', 'green', 'red', 'navy', 'forestgreen', 'crimson',
#         'darkviolet', 'teal', 'goldenrod', 'indigo', 'olive', 'maroon'
#     ]
#     color_cycle = (color for color in colors)
#     # Markers for each type of Y
#     markers = ['o' for i in range(12)]  # Markers for Y1, Y2, Y3
#
#     # Loop through each dataset
#     for dataset_index, dataset in enumerate(data_list):
#         df = pd.DataFrame(dataset)
#         color = colors[dataset_index % len(colors)]
#         for y_index, y_col in enumerate(['Y1', 'Y2', 'Y3']):
#             marker = markers[y_index]
#             label = labels[dataset_index][y_index]
#             # ax.plot(df["X"], df[y_col], marker=marker+'-', color=color, label=label)
#             ax.plot(df["X"], df[y_col], marker=marker, linestyle="-", color=next(color_cycle),
#                     label=label)
#     # Setting titles and labels
#     # ax.set_title('Simulated Dataset')
#     # ax.set_xlabel('X Axis (0 to 1)')
#     ax.set_ylabel('Youden\'s index')
#     ax.grid(True)
#     ax.legend(loc = "upper right",fontsize=6, ncol=1)




if __name__=="__main__":
    # Define the data for each dataset
    data1 = {
        "X": [0.039231077, 0.07001889, 0.133974596, 0.209430585, 0.367544468],
        "Y1": [0.979304076, 0.919344924, 0.886044521, 0.855831431, 0.807674437],
        "Y2": [0.975879418, 0.912772216, 0.775466134, 0.862911774, 0.807129696],
        "Y3": [0.960616438, 0.912460485, 0.879204718, 0.869203903, 0.807402067]
    }
    data2 = {
        "X": [0.054094697, 0.133974596, 0.233035011, 0.367544468, 0.422649731],
        "Y1": [0.929745597, 0.943333822, 0.887335756, 0.846575342, 0.790555155],
        "Y2": [0.92778865, 0.949954216, 0.886607419, 0.819726027, 0.794655732],
        "Y3": [0.921318493, 0.957032452, 0.878446115, 0.83, 0.790915645]
    }
    data3 = {
        "X": [0.036375888, 0.147197135, 0.233035011, 0.3827866, 0.445299804],
        "Y1": [0.952054795, 0.976788432, 0.920648, 0.878543203, 0.783442053],
        "Y2": [0.938356164, 0.968797565, 0.919336011, 0.875500527, 0.810013046],
        "Y3": [0.945205479, 0.95281583, 0.910105499, 0.87664647, 0.807751685]
    }
    data4 = {
        "X": [0.014389239, 0.133974596, 0.257218647, 0.3827866, 0.445299804],
        "Y1": [0.864726027, 0.91957732, 0.892220914, 0.871667545, 0.795596869],
        "Y2": [0.852739726, 0.904966669, 0.897700366, 0.853345627, 0.795564253],
        "Y3": [0.871575342, 0.916612336, 0.896770594, 0.839225501, 0.796129593]
    }

    # Convert dictionaries to DataFrames
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df4 = pd.DataFrame(data4)

    # Create a plot
    plt.figure(figsize=(12, 7))

    # Plot each dataset with unique color and marker
    plt.plot(df1["X"], df1["Y1"], 'o-', color='red', label='Data 1 - Y1')
    plt.plot(df1["X"], df1["Y2"], 's-', color='red', label='Data 1 - Y2')
    plt.plot(df1["X"], df1["Y3"], '^-', color='red', label='Data 1 - Y3')

    plt.plot(df2["X"], df2["Y1"], 'o-', color='blue', label='Data 2 - Y1')
    plt.plot(df2["X"], df2["Y2"], 's-', color='blue', label='Data 2 - Y2')
    plt.plot(df2["X"], df2["Y3"], '^-', color='blue', label='Data 2 - Y3')

    plt.plot(df3["X"], df3["Y1"], 'o-', color='green', label='Data 3 - Y1')
    plt.plot(df3["X"], df3["Y2"], 's-', color='green', label='Data 3 - Y2')
    plt.plot(df3["X"], df3["Y3"], '^-', color='green', label='Data 3 - Y3')

    plt.plot(df4["X"], df4["Y1"], 'o-', color='purple', label='Data 4 - Y1')
    plt.plot(df4["X"], df4["Y2"], 's-', color='purple', label='Data 4 - Y2')
    plt.plot(df4["X"], df4["Y3"], '^-', color='purple', label='Data 4 - Y3')

    plt.title('Combined Point-Line Plot')
    plt.xlabel('X')
    plt.ylabel('Y values')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()
