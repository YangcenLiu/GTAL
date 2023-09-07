import json
from collections import defaultdict
from pathlib import Path
import pickle
import random
from rich import print
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F

import models
import single_stream_model
import options
import proposal_methods as PM
import utils.wsad_utils as utils
import wsad_dataset
from eval.classificationMAP import getClassificationMAP as cmAP
from eval.eval_detection import ANETdetection

from time import time
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    '''
    for i in range(data.shape[0]):
        if int(label[i]) == 1:
            s1 = plt.scatter(data[i, 0], data[i, 1], c="orange", s=0.2) 
        elif int(label[i]) == 2:
            s2 = plt.scatter(data[i, 0], data[i, 1], c="brown", s=0.2)
        elif int(label[i]) == 3:
            s3 = plt.scatter(data[i, 0], data[i, 1], c="magenta", s=0.2)
        elif int(label[i]) == 4:
            s4 = plt.scatter(data[i, 0], data[i, 1], c="blue", s=0.2)
        elif int(label[i]) == 5:
            s5 = plt.scatter(data[i, 0], data[i, 1], c="purple", s=0.2)
        elif int(label[i]) == 6:
            s6 = plt.scatter(data[i, 0], data[i, 1], c="black", s=0.2)
        elif int(label[i]) == 7:
            s7 = plt.scatter(data[i, 0], data[i, 1], c="midnightblue", s=0.2)
        elif int(label[i]) == 8:
            s8 = plt.scatter(data[i, 0], data[i, 1], c="grey", s=0.2)
        elif int(label[i]) == 9:
            s9 = plt.scatter(data[i, 0], data[i, 1], c="red", s=0.2)
        elif int(label[i]) == 10:
            s10 = plt.scatter(data[i, 0], data[i, 1], c="green", s=0.2)
    '''

    for i in range(data.shape[0]):
        if int(label[i]) == 1:
            s1 = plt.scatter(data[i, 0], data[i, 1], c="red", s=0.2) 
        elif int(label[i]) == 2:
            s2 = plt.scatter(data[i, 0], data[i, 1], c="green", s=0.2)
        elif int(label[i]) == 3:
            s3 = plt.scatter(data[i, 0], data[i, 1], c="magenta", s=0.2)
        elif int(label[i]) == 4:
            s4 = plt.scatter(data[i, 0], data[i, 1], c="blue", s=0.2)

    plt.xticks([])
    plt.yticks([])
    
    plt.legend((s1,s2,s3,s4),('ThumosBackground','ThumosForeground','AnetBackground','AnetForeground'),
        loc = 'best', prop = {'size':6})
    plt.title(title)
    return fig

def visualise(features, labels):
    # data, label, n_samples, n_features = get_data()

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    result = tsne.fit_transform(features)
    print('result.shape',result.shape)
    fig = plot_embedding(result, labels,
                         't-SNE embedding of Video features for background and foreground')
    plt.savefig("AnetThumosForeground.png")


if __name__ == '__main__':

    Anetfeature = np.load("Anet1.2features.npy")
    Anetlabels = np.load("Anet1.2labels.npy")

    Thumosfeature = np.load("Thumos14features.npy")
    Thumoslabels = np.load("Thumos14labels.npy")
    
    all_features = []
    all_labels = []

    for i in range(len(Thumosfeature)):
        all_features.append(Thumosfeature[i])
        if Anetlabels[i] in [0]:
            all_labels.append(1)
        else:
            all_labels.append(2)

    for i in range(len(Anetfeature)):
        all_features.append(Anetfeature[i])
        if Anetlabels[i] in [0]:
            all_labels.append(3)
        else:
            all_labels.append(4)
    
   
    visualise(np.array(all_features), np.array(all_labels))


