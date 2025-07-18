import numpy as np
import matplotlib
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime



def get_color(data):
    print(data.shape)
    #return (data[-7].tolist(),data[-6].tolist(), data[-5].tolist())
    idx_for_shape = [data[-8], data[-7],data[-6]]
    max_value = max(idx_for_shape)
    max_index = idx_for_shape. index(max_value)
    color_list = ['r','y','b']
    color  = color_list[max_index]

    return color

def get_shape(data):
    idx_for_shape = [data[-5], data[-4],data[-3]]
    max_value = max(idx_for_shape)
    max_index = idx_for_shape. index(max_value)
    shape_list = ['techteckig','circle','triangle']
    shape  = shape_list[max_index]

    return shape
#    if max_index ==1:
#        return [[data[0], data[2]],[data[0], data[3]], [data[1], data[2]], [data[1], data[3]]]
#    if max_index ==2:
#        return [[data[0],data[2]], [data[0], data[2]], [0.5*(data[0]+data[1]), data[3]]]
#    else:
#        return

def draw_kandinsky(data,id,img_id, folder):
    data = data[0]
    'adjust axes_y'
   # data[:,1] = 1- data [:,1]
   # data[:,3] = 1- data[:,3]

    plt.figure(figsize = (15, 15))
    plt.ylim([1, 0])
    for i in range(data.shape[0]):
        color = get_color(data[i])
        shape = get_shape(data[i])
        if shape =='circle':
            t1 = plt.Circle(( data[i][0] + 0.5*(data[i][2]-data[i][0]) , data[i][1] + 0.5*(data[i][3]-data[i][1]) ), 0.5*(data[i][3]-data[i][1]) , color = color)
            plt.gca().add_patch(t1)
        elif shape =='triangle':
            t2 = plt.Polygon([[data[i][0],data[i][3]], [data[i][2], data[i][3]], [data[i][0] + 0.5*(data[i][2]-data[i][0]),data[i][1]]], color = color)
            plt.gca().add_patch(t2)
        else:
            t3 = patches.Rectangle((data[i][0], data[i][1]), data[i][2]-data[i][0], data[i][3]-data[i][1], color =color)
           # t3 = plt.Polygon([[data[i][1],data[i][3]], [data[i][0], data[i][3]], [data[i][1] + 0.5*(data[i][0]-data[i][1]), data[i][2]]], color=color)
            plt.gca().add_patch(t3)
    #plt.savefig('result/'+str(id)+str(img_id))
    plt.savefig('E:/NSFR-Planner/NSFR-Planner/result/kandinsky2/' +folder+str(img_id)+str(id) +datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.png')
    return 0
