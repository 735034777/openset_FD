# Define a function for polynomial fitting
import pandas as pd
import matplotlib.pyplot as plt

def draw_scatter(data):
    def poly_func(x, a, b, c):
        return a * x ** 2 + b * x + c
    new_data = data

    # Prepare data for plotting and fitting
    x_data = new_data['openness']
    metrics = ['recall', 'f1', 'youdens_index']
    colors = ['blue', 'red', 'green']
    fit_params = {}

    plt.figure(figsize=(10, 6))

    # Fit and plot for each metric
    for metric, color in zip(metrics, colors):
        y_data = new_data[metric]
        params, _ = curve_fit(poly_func, x_data, y_data)
        fit_params[metric] = params
        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        y_fit = poly_func(x_fit, *params)

        # Plot original data and fitted curve
        plt.scatter(x_data, y_data, color=color, label=f'{metric} Data')
        plt.plot(x_fit, y_fit, color=color, linestyle='--', label=f'{metric} Fit')

    plt.xlabel('Openness')
    plt.ylabel('Metric Values')
    plt.title('Metric Analysis with Polynomial Fit')
    plt.legend()
    plt.grid(True)
    plt.show()

fit_params
