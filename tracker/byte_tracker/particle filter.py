# -*- coding: utf-8 -*-

"""
粒子滤波
10-04
zlw
"""
from random import random

import numpy as np
import scipy
from numpy.random import uniform, randn


'''
初始化粒子
'''
def create_uniform_particles(x_range, y_range, hdg_range, N):
    # （x,y,航向）
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles


def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles


print(create_uniform_particles((0,1), (0,1), (0, np.pi*2), 100))

'''
粒子里添加输入噪声
'''
def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`
    通过噪声Q 控制输入u """

    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist


'''
粒子的更新，后验估计
'''

def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        # 归一化 粒子们与测量值 的距离
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)  # 每个行向量求二范数（平方和开根号）
        weights *= scipy.stats.norm(distance, R).pdf(z[i])  # R不确定数sigma**2，z landmark是啥？ 高斯分布

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize


'''
计算状态估计
返回加权粒子的均值和方差
'''
def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var


particles = create_uniform_particles((0, 1), (0, 1), (0, 5), 1000)
weights = np.array([.25]*1000)
estimate(particles, weights)

'''
粒子重采样: 重采样可以丢弃概率极低的粒子，取而代之的是概率高的粒子的重复值带噪声
'''
def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)

'''
need to create the particles and the landmarks
执行一个循环，依次调用预测、更新、重采样，然后用估计计算新的状态估计。
'''


