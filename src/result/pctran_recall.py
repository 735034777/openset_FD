import matplotlib.pyplot as plt
import pandas as pd


def plot_pctran_recall(ax):
    # Data definitions
    # Data for each dataset
    data1 = {
        "X": [0.039231077, 0.133974596, 0.244071053981545, 0.340619526604213, 0.434314575050762],
        "Y1": [0.97869101978691, 0.944063926940639, 0.928082191780822, 0.920126448893572, 0.90914585012087],
        "Y2": [0.97869102, 0.93935502283105, 0.920662100456621, 0.920547945205479, 0.932715551974214],
        "Y3": [0.964992389649924, 0.947916666666666, 0.912100456621004, 0.929820864067439, 0.922240128928283]
    }

    data2 = {
        "X": [0.054094697, 0.133974596, 0.233035011, 0.367544468, 0.422649731],
        "Y1": [0.940068493, 0.944063927, 0.925496226, 0.913926941, 0.88630137],
        "Y2": [0.938356164, 0.939355023, 0.92507688, 0.893150685, 0.888527397],
        "Y3": [0.932791096, 0.947916667, 0.917808219, 0.901369863, 0.885616438]
    }

    data3 = {
        "X": [0.036375888, 0.147197135, 0.233035011, 0.3827866, 0.445299804],
        "Y1": [0.96803653, 0.9847793, 0.946603299, 0.932139094, 0.882496195],
        "Y2": [0.95890411, 0.98021309, 0.945205479, 0.929610116, 0.896803653],
        "Y3": [0.96347032, 0.97108067, 0.936678781, 0.93024236, 0.895585997]
    }

    data4 = {
        "X": [0.014389239, 0.133974596, 0.257218647, 0.3827866, 0.445299804],
        "Y1": [0.879756469, 0.944063927, 0.928696874, 0.928345627, 0.889041096],
        "Y2": [0.869101979, 0.939355023, 0.933263084, 0.917386723, 0.888127854],
        "Y3": [0.885844749, 0.947916667, 0.932736214, 0.908746048, 0.888432268]
    }

    # Additional data sets can be defined similarly...

    # Plotting styles, colors, and labels
    # styles = ['o-', 's-', '^-', 'o--', 's--', '^--', 'o-.', 's-.', '^-.', 'o:', 's:', '^:']
    styles = ['o-' for i in range(12)]
    colors = [
        'blue', 'green', 'red', 'navy', 'forestgreen', 'crimson',
        'darkviolet', 'teal', 'goldenrod', 'indigo', 'olive', 'maroon'
    ]
    labels = [
        'SDOS cosine', 'SDOS euclidean', 'SDOS mahalanobis',
        '50%->100% cosine', '50%->100% euclidean', '50%->100% mahalanobis',
        '50%->70% cosine', '50%->70% euclidean', '50%->70% mahalanobis',
        '70%->100% cosine', '70%->100% euclidean', '70%->100% mahalanobis',
    ]

    # Dataset list
    datasets = [data1, data2,data3,data4]  # Add additional datasets to this list

    # Loop through the datasets and plot each one
    for data_dict, i in zip(datasets, range(0, len(styles), 3)):
        data = pd.DataFrame(data_dict)
        for j in range(3):  # Assuming three Y values per dataset
            ax.plot(data["X"], data[f"Y{j+1}"], styles[i+j], color=colors[i+j], label=labels[i+j])

    # Set titles and labels
    # ax.set_title('Simulated Dataset')
    # ax.set_xlabel('X Axis (0 to 1)')
    # ax.set_ylabel('Recall')
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(True)
    ax.legend(loc = "upper right",fontsize=7, ncol=1)
    ax.text(0.01, 0.9, '(d)', transform=ax.transAxes, size=15, weight='bold')



if __name__=="__main__":
    # Data for each dataset
    data1 = {
        "X": [0.039231077, 0.07001889, 0.133974596, 0.209430585, 0.367544468],
        "Y1": [0.98173516, 0.930453109, 0.917636986, 0.905964612, 0.891019787],
        "Y2": [0.97869102, 0.924341412, 0.817636986, 0.909674658, 0.890715373],
        "Y3": [0.96499239, 0.923919916, 0.91130137, 0.914526256, 0.89086758]
    }
    data2 = {
        "X": [0.054094697, 0.133974596, 0.233035011, 0.367544468, 0.422649731],
        "Y1": [0.940068493, 0.952366127, 0.925496226, 0.913926941, 0.88630137],
        "Y2": [0.938356164, 0.958437111, 0.92507688, 0.893150685, 0.888527397],
        "Y3": [0.932791096, 0.964819427, 0.917808219, 0.901369863, 0.885616438]
    }
    data3 = {
        "X": [0.036375888, 0.147197135, 0.233035011, 0.3827866, 0.445299804],
        "Y1": [0.96803653, 0.9847793, 0.946603299, 0.932139094, 0.882496195],
        "Y2": [0.95890411, 0.98021309, 0.945205479, 0.929610116, 0.896803653],
        "Y3": [0.96347032, 0.97108067, 0.936678781, 0.93024236, 0.895585997]
    }
    data4 = {
        "X": [0.014389239, 0.133974596, 0.257218647, 0.3827866, 0.445299804],
        "Y1": [0.879756469, 0.931818182, 0.928696874, 0.928345627, 0.889041096],
        "Y2": [0.869101979, 0.918430884, 0.933263084, 0.917386723, 0.888127854],
        "Y3": [0.885844749, 0.928860523, 0.932736214, 0.908746048, 0.888432268]
    }

    # Convert data to DataFrames
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df4 = pd.DataFrame(data4)

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(df1["X"], df1["Y1"], 'o-', color="blue", label="Original Y1")
    plt.plot(df1["X"], df1["Y2"], 's-', color="green", label="Original Y2")
    plt.plot(df1["X"], df1["Y3"], '^-', color="red", label="Original Y3")
    plt.plot(df2["X"], df2["Y1"], 'o--', color="cyan", label="New Y1")
    plt.plot(df2["X"], df2["Y2"], 's--', color="magenta", label="New Y2")
    plt.plot(df2["X"], df2["Y3"], '^--', color="yellow", label="New Y3")
    plt.plot(df3["X"], df3["Y1"], 'o-.', color="purple", label="Latest Y1")
    plt.plot(df3["X"], df3["Y2"], 's-.', color="orange", label="Latest Y2")
    plt.plot(df3["X"], df3["Y3"], '^-.', color="lime", label="Latest Y3")
    plt.plot(df4["X"], df4["Y1"], 'o:', color="black", label="Newer Y1")
    plt.plot(df4["X"], df4["Y2"], 's:', color="grey", label="Newer Y2")
    plt.plot(df4["X"], df4["Y3"], '^:', color="brown", label="Newer Y3")
    plt.title("Comparison of Data Sets")
    plt.xlabel("X")
    plt.ylabel("Y-Values")
    plt.legend()
    plt.grid(True)
    plt.show()
