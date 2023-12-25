import numpy as np
from scipy.stats import gaussian_kde


def characteristic_function(data, point):
    """
    Calculate the characteristic function for a given data set at a specified point.
    """
    data = np.asarray(data)
    point = np.asarray(point)
    return np.mean(np.exp(1j * data.dot(point)))


def gini_distance_covariance(X, Y):
    """
    Compute the Gini distance covariance between two datasets X and Y.
    """
    n = len(X)
    if len(Y) != n:
        raise ValueError("X and Y must have the same length")

    # Estimate the characteristic functions
    cov = 0
    for x in X:
        for y in Y:
            phi_x = characteristic_function(X, x)
            phi_y = characteristic_function(Y, y)
            cov += np.abs(phi_x - phi_y)**2

    return cov / n**2

def gini_distance_correlation(X, Y):
    """
    Compute the Gini distance correlation between two datasets X and Y.
    """
    g_cov_xy = gini_distance_covariance(X, Y)
    g_cov_xx = gini_distance_covariance(X, X)
    g_cov_yy = gini_distance_covariance(Y, Y)

    return g_cov_xy / np.sqrt(g_cov_xx * g_cov_yy)


if __name__=="__main__":
    # 示例数据
    X = np.random.normal(0, 1, 100)
    Y = np.random.normal(0, 1, 100)

    # 计算基尼距离协方差
    g_cov = gini_distance_covariance(X, Y)
    print(f"Gini Distance Covariance: {g_cov}")

    # 计算基尼距离相关性
    g_cor = gini_distance_correlation(X, Y)
    print(f"Gini Distance Correlation: {g_cor}")
