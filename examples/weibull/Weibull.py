"""
File: Weibull
Author: admin
Date Created: 2024/6/28
Last Modified: 2024/6/28

Description:
    存储weibull类


"""

import numpy as np

class Weibull:
    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def pdf(self, x):
        return (self.shape / self.scale) * (x / self.scale) ** (self.shape - 1) * np.exp(-(x / self.scale) ** self.shape)


    def cdf(self, x):
        return 1 - np.exp(-(x / self.scale) ** self.shape)

