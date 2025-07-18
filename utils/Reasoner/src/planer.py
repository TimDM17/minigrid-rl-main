import numpy as np
import matplotlib
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Kandinsky_plot import  draw_kandinsky




def move_objects(data,n):
    'input:'
    'data: extracts object information: x1x2y1y2+RGB+Shape'
    'n: object that need to be moved'
    data = data[0]
    data[n][0] += 0.05
    data[n][2] += 0.05
    return data

def middle_line (data):
    data = data[0]
    initial_point = 0
    for i in range(data.shape[0]):
        width = data[i][2] - data[i][0]
        data[i][0] = initial_point
        data[i][2] = initial_point + width
        length = data[i][3] - data[i][1]
        data [i][1]= 0.5 - 0.5*length
        data[i][3] = 0.5 + 0.5*length
        initial_point = data[i][2] + 0.05
    return data

def vertical_line (data):
    data = data[0]
    initial_point = 0
    for i in range(data.shape[0]):
        width = data[i][1] - data[i][3]
        data[i][3] = initial_point
        data[i][1] = initial_point + width
        length = data[i][2] - data[i][0]
        data [i][0]= 0.5 - 0.5*length
        data[i][2] = 0.5 + 0.5*length
        initial_point = data[i][1] + 0.05
    return data


