#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

from turtle import color
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

def load(policy, num=3):
    url = os.path.dirname(os.path.realpath(__file__))
    result = []
    for i in range(num):
        # temp = np.load(url + "\\" + policy + "\\train_7m_seed_" + str(i+1) + "\\" + "step_rewards.npy")
        temp = np.load(url + "/" + policy  + str(i+1) + "_" + "step_rewards.npy")

        result.append(temp)
    return result

def smooth(arr, fineness):
    result = arr[:]
    for i in range(fineness, arr.size):
        temp = 0
        for j in range(fineness):
            temp += result[i-j]
        result[i] = temp/fineness
    return np.array(result)

def get_mean_max_min(data_list, smooth_flag, fineness):
    n = sys.maxsize
    for data in data_list:
        n = min(n, data.size)
    max_data = np.zeros((n))
    min_data = np.zeros((n))
    mean_data = np.zeros((n))
    for i in range(n):
        temp = []
        for data in data_list:
            temp.append(data[i])
        temp = np.array(temp)
        max_data[i] = temp.max()
        min_data[i] = temp.min()
        mean_data[i] = temp.mean()

    data = [mean_data, max_data, min_data]
    if smooth_flag:
        for i in range(len(data)):
            for j in range(2, fineness):
                data[i] = smooth(data[i], j)
    return data[0][1:], data[1][1:], data[2][1:]

if __name__ == "__main__":
    # SAC_data = load("SAC", 3)
    TD3_data = load("TD3", 3)
    DDPG_data = load("DDPG", 3)

    fineness = 20

    # SAC_mean_data, SAC_max_data, SAC_min_data = get_mean_max_min(SAC_data, True, fineness)
    TD3_mean_data, TD3_max_data, TD3_min_data = get_mean_max_min(TD3_data, True, fineness)
    DDPG_mean_data, DDPG_max_data, DDPG_min_data = get_mean_max_min(DDPG_data, True, fineness)

    n =min(TD3_mean_data.size, DDPG_mean_data.size)
    # n = min(SAC_mean_data.size, TD3_mean_data.size)

    # SAC_x = range(SAC_mean_data.size)
    # plt.fill_between(SAC_x, SAC_min_data, SAC_max_data, alpha=0.2)
    # plt.plot(SAC_x, SAC_mean_data, linewidth=2, label="SAC + PER")

    TD3_x = range(TD3_mean_data.size)
    plt.fill_between(TD3_x, TD3_min_data, TD3_max_data, alpha=0.2, color="g")
    plt.plot(TD3_x, TD3_mean_data, linewidth=2, label="TD3", color="g")

    DDPG_x = range(DDPG_mean_data.size)
    plt.fill_between(DDPG_x, DDPG_min_data, DDPG_max_data, alpha=0.2, color="r")
    plt.plot(DDPG_x, DDPG_mean_data, linewidth=2, label="DDPG", color="r")

    ax = plt.gca()
    ax.set_xlim(0, n)
    # ax.set_ylim(-1.0, 1.7)

    plt.grid(linestyle='-.')

    plt.title("Training step-reward curve of DDPG and TD3",fontsize=15)
    plt.xlabel("Step", labelpad=10,fontsize=15)
    plt.ylabel("Reward", labelpad=10,fontsize=15)

    plt.legend(loc="lower right", frameon=False)
    plt.savefig('step_reward_plot.png',dpi=300)
    plt.show()
