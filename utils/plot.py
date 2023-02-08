import math, random

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


def plot_vanilla(data_list, min_len):

    sns.set_style("whitegrid", {'axes.grid' : True,'axes.edgecolor':'black'})
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()
    colors = ["red", "black", "green", "blue", "purple",  "darkcyan", "brown", "darkblue",]
    labels = ["DQN", "DDQN","Maxmin", "EnsembleDQN", "MaxminDQN"]
    color_patch = []
    for color, label, data in zip(colors, labels, data_list):
        sns.lineplot(time=range(min_len), data=data, color=color, ci=95)
        color_patch.append(mpatches.Patch(color=color, label=label))
    print(min_len)
    plt.xlim([0, min_len])
    plt.xlabel('Training Episodes $(\\times10^6)$', fontsize=22)
    plt.ylabel('Average return', fontsize=22)
    lgd=plt.legend(
    frameon=True, fancybox=True, \
    prop={'weight':'bold', 'size':14}, handles=color_patch, loc="best")
    plt.title('Title', fontsize=14)
    ax = plt.gca()
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_single(data_list, min_len, file_name):

    sns.set_style("whitegrid", {'axes.grid' : True,'axes.edgecolor':'black'})
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()
    colors = ["red", "black", "green", "blue", "purple",  "darkcyan", "brown", "darkblue",]
    labels = ["DDQN"]
    color_patch = []

    for color, label, data in zip(colors, labels, data_list):
        sns.lineplot(x = data[0], y = data[1], color=color, ci=95)#, err_style = "band"
        color_patch.append(mpatches.Patch(color=color, label=label))
    
    plt.xlim([0, min_len])
    plt.xlabel('Training Episodes', fontsize=16)
    plt.ylabel('Average return', fontsize=16)
    #lgd=plt.legend(frameon=True, fancybox=True, prop={'weight':'bold', 'size':14}, handles=color_patch, loc="best")
    #plt.title('Title', fontsize=14)
    ax = plt.gca()
    #ax.set_xticks([10, 20, 30, 40, 50])
    #ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.show()
    plt.savefig('./'+file_name)
    plt.close()

def plot_array(data_list, min_len, file_name, x_name, y_name):

    sns.set_style("whitegrid", {'axes.grid' : True,'axes.edgecolor':'black'})
    fig = plt.figure(figsize=(8,8))
    plt.clf()
    ax = fig.gca()
    colors = ["red", "black", "green", "blue", "purple",  "darkcyan", "brown", "darkblue",]
    labels = ["DDQN"]
    color_patch = []

    for color, label, data in zip(colors, labels, data_list):
        sns.lineplot(x = data[0], y = data[1], color=color, ci=95)#, err_style = "band"
        color_patch.append(mpatches.Patch(color=color, label=label))
    
    plt.xlim([0, min_len])
    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    lgd=plt.legend(
    frameon=True, fancybox=True, \
    prop={'weight':'bold', 'size':14}, handles=color_patch, loc="best")
    plt.title('Title', fontsize=14)
    ax = plt.gca()
    #ax.set_xticks([10, 20, 30, 40, 50])
    #ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.show()
    plt.savefig('./'+file_name)
    plt.close()

def plot(frame_idx, rewards, losses, file_name):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()
    plt.savefig('./'+file_name)
    plt.close()

