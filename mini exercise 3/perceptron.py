"""
590D: Mini Excerise 3
Submitted by -
    Kriti Shrivastava
    Nidhi Mundra
    Utkarsh Srivastava
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
        # print ("Point : ", index)
        # print (np.sum(a[index]*w)*l[index])
        if np.sum(a[index]*w)*l[index] <= 0:
            return False
    return True



def plot_graphs(data, succesive_weights, a):
    """
    :param data: Points data
    :param succesive_weights: Successive weight vectors
    :return: Plot scatterplot for points and weights
    """
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    # don't plot the 3rd bias dimension

    # plot points (normalized)    
    for index in range(a.shape[0]):
        if data[index][2] == 1:
            x1.append(a[index][0])
            y1.append(a[index][1])
        else:
            x2.append(a[index][0])
            y2.append(a[index][1])

    # plot each successive weight line
    random_pts = np.linspace(-1.0, 1.0)
    lines = []
    cmap = plt.cm.get_cmap('hsv', len(succesive_weights)-1)
    for idx, weight in enumerate(succesive_weights):
        m1 = -weight[0]/weight[1]
        m2 = -weight[2]/weight[1]
        if idx == len(succesive_weights)-1:
            lines.append((plt.plot(random_pts, m1 * random_pts + m2, 'r', label ='Final W'))[0])
        else:
            lines.append((plt.plot(random_pts, m1 * random_pts + m2, linestyle='--', linewidth=0.5, color=cmap(idx), label='W@t='+ str(idx)))[0])
    plt.scatter(x1, y1, color='b')
    plt.scatter(x2, y2, color = 'g')
    plt.xlabel('1st dimension')
    plt.ylabel('2nd dimension')
    plt.title('Perceptron learning @ each iteration')
    imagename = "figure_n=" + str(len(data)) + ".png"
    plt.legend(handles=lines)
    plt.savefig(imagename)
    plt.close()

def cal_margin(a, w, l):
    """
    :param a: Points
    :param w: Final weight
    :return: Prints margin
    """
    # margin = (a.w)/|w| (where a is already normalized)
    # includes the bias term in margin computation

    denom = np.sqrt(np.sum(w**2))
    min_val = sys.float_info.max
    for index in range(a.shape[0]):
        margin = (np.sum(a[index] * w)*l[index])/ denom
        if margin < min_val:
            min_val = margin
    print "Margin distance : ", margin

def perceptron(data):
    """
    :param data: Points data
    :return: Runs perceptron learning algorithm, calculates: weight, threshold and margin
    """
    a = np.zeros((len(data), 3))
    l = np.zeros((len(data),))
    w = np.zeros((3,))
    succesive_weights = []

    # add bias term
    for idx, point in enumerate(data):
        a[idx][0] = point[0]*1.0
        a[idx][1] = point[1]*1.0
        a[idx][2] = 1
        l[idx] = point[2]

    # normalize points
    for idx in range(a.shape[0]):
        a[idx] = a[idx]/np.sqrt(np.sum(a[idx]**2))
    
    # Initialize weight (a1.l1)
    w = np.reshape(a[0]*l[0],(3,))
    succesive_weights.append(np.copy(w))


    max_iters = 10000
    counter = 0
    # perceptron algo
    more_iter = True
    while(more_iter and counter<max_iters):
    
        # find some point which mis-classifies
        index = random.randint(0, len(data)-1)
        while(np.sum(a[index]*w)*l[index] > 0):
            index = random.randint(0, len(data)-1)

        # update weight
        w += np.reshape(a[index]*l[index],(3,))
        succesive_weights.append(np.copy(w))

        # check terminating condition
        if(all_points_separated(a, l, w) == True):
            more_iter = False

        counter+=1


    print "Full Weight list per iteration : ", succesive_weights
    
    # plotting graph
    plot_graphs(data, succesive_weights, a)
    # find margin
    cal_margin(a, w, l)


# produce the experiment for n=4,10,20
counts = [2, 5, 10]
for n in counts:
    print "Experiment for n = ", n
    data = generateData(n)
    perceptron(data)
    print ''
