"""
590D: Mini Excerise 3
Submitted by -
    Utkarsh Srivastava
    Nidhi Mundra
    Kriti Shrivastava
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import sys

#generates a vector of c*2 random points and labels.
def generateData(c):
    xa = []
    ya = []
    xb = []
    yb = []
    for i in range(c):
        xa.append((random.random()*2-1)/2-0.5)
        ya.append((random.random()*2-1)/2+0.5)
        xb.append((random.random()*2-1)/2+0.5)
        yb.append((random.random()*2-1)/2-0.5)
    data = []
    for i in range(len(xb)):
        data.append([xa[i],ya[i],1])
        data.append([xb[i],yb[i],-1])
    return data

def all_points_separated(a, l, w):
    """
    :param a: Points to be separated
    :param l: Labels
    :param w: Weight
    :return: True if all points are linearly separated, false otherwise
    """
    for index in range(a.shape[0]):
        print ("Point : ", index)
        print (np.sum(a[index]*w)*l[index])
        print (l[index])
        if np.sum(a[index]*w)*l[index] <= 0:
            print("a")
            return False
    return True


def plot_graphs(data, succesive_weights):
    """
    :param data: Points data
    :param succesive_weights: Successive weight vectors
    :return: Plot scatterplot for points and weights
    """
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for point in data:
        if point[2] == 1:
            x1.append(point[0])
            y1.append(point[1])
        else:
            x2.append(point[0])
            y2.append(point[1])
    w_x = []
    w_y = []
    for weight in succesive_weights:
        w_x.append(weight[0])
        w_y.append(weight[1])
    plt.scatter(x1, y1, color='b')
    plt.scatter(x2, y2, color = 'g')
    plt.scatter(w_x, w_y, color='r')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Perceptron')
    imagename = "figure_" + str(len(data)) + ".png"
    plt.savefig(imagename)
    plt.close()

def cal_margin(a, w):
    """
    :param a: Points
    :param w: Final weight
    :return: Prints margin
    """
    deno = np.sqrt(np.sum(w**2))
    min = sys.float_info.max
    for index in range(a.shape[0]):
        margin = np.sum(a[index] * w)/ deno
        if margin < min:
            min = margin
    print(margin)

def perceptron(data):
    """
    :param data: Points data
    :return: Runs perceptron learning algorithm, calculates: weight, threshold and margin
    """
    a = np.zeros((len(data), 3))
    l = np.zeros((len(data),))
    w = np.zeros((3,))
    succesive_weights = []
    for idx, point in enumerate(data):
        a[idx][0] = point[0]*1.0
        a[idx][1] = point[1]*1.0
        a[idx][2] = 1
        l[idx] = point[2]
    # Initialize weight
    w = np.reshape(a[0]*l[0],(3,))
    succesive_weights.append(np.copy(w))
    more_iter = True
    while(more_iter):
        index = random.randint(0, len(data)-1)
        while(np.sum(a[index]*w)*l[index] > 0):
            index = random.randint(0, len(data)-1)
            # print(np.sum(a[index]*w)*l[index])
        w += np.reshape(a[index]*l[index],(3,))
        succesive_weights.append(np.copy(w))
        if(all_points_separated(a, l, w) == True):
            more_iter = False
    print(succesive_weights)
    plot_graphs(data, succesive_weights)
    cal_margin(a, w)

counts = [2, 5, 10]
for n in counts:
    data = generateData(n)
    print(data)
    perceptron(data)