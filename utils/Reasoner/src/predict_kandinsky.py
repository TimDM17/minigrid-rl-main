import argparse
import random
import numpy as np
import copy

import torch
import torch.nn as nn
import math


from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from PIL import Image



from .nsfr_utils import denormalize_kandinsky, get_data_loader, get_prob, get_nsfr_model
from .nsfr_utils import save_images_with_captions, to_plot_images_kandinsky, generate_captions
from .logic_utils import get_lang, get_mi_lang




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--e", type=int, default=4,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", default="two_doors", choices=["two_doors", "one_door"], help="Use kandinsky patterns dataset")
    parser.add_argument("--dataset-type", default="DoorKey",
                        help="kandinsky or clevr")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", default="True",action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", default= "True",action="store_true",
                        help="Plot images with captions.")
    args = parser.parse_args()
    return args

def initialize_reasoner(door_type):

    args = get_args()
    args.dataset = door_type


    if args.no_cuda:
        device = torch.device('cpu')
    elif len(args.device.split(',')) > 1:
        # multi gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:' + args.device)

    #run_name = 'predict/' + args.dataset



    # load logical representations
    lark_path = '/Users/yezihan/Desktop/minigrid/rl-starter-files_roomSetting/utils/Reasoner/src/lark/exp.lark'
    lang_base_path = '/Users/yezihan/Desktop/minigrid/rl-starter-files_roomSetting/utils/Reasoner/data/lang/'
    lang, clauses, atoms = get_lang(lark_path, lang_base_path, args.dataset_type, args.dataset)

    bk= []

    lang_mi, clauses_mi, atoms_mi = get_mi_lang(clauses, atoms, lark_path, lang_base_path, args.dataset_type, args.dataset)

    # Neuro-Symbolic Forward Reasoner
    NSFR = get_nsfr_model(args, lang, clauses, atoms, bk, device,atoms_mi, clauses_mi, lang_mi)

    return NSFR

def initialize_percept_reasoner(door_type):

    args = get_args()
    args.dataset = door_type
    args.dataset_type = "DoorKey_Step"


    if args.no_cuda:
        device = torch.device('cpu')
    elif len(args.device.split(',')) > 1:
        # multi gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:' + args.device)

    #run_name = 'predict/' + args.dataset



    # load logical representations
    lark_path = '/Users/yezihan/Desktop/minigrid/rl-starter-files_roomSetting/utils/Reasoner/src/lark/exp.lark'
    lang_base_path = '/Users/yezihan/Desktop/minigrid/rl-starter-files_roomSetting/utils/Reasoner/data/lang/'
    lang, clauses, atoms = get_lang(lark_path, lang_base_path, args.dataset_type, args.dataset)

    bk= []

    lang_mi, clauses_mi, atoms_mi = get_mi_lang(clauses, atoms, lark_path, lang_base_path, args.dataset_type, args.dataset)

    # Neuro-Symbolic Forward Reasoner
    NSFR = get_nsfr_model(args, lang, clauses, atoms, bk, device,atoms_mi, clauses_mi, lang_mi)

    return NSFR

def predict_reward(NSFR, symbol_state):


    V_0_mi, V_T_mi, atoms_mi = NSFR(symbol_state)



    return V_T_mi, atoms_mi

def predict_step_reward(NSFR, NSFR_step, symbol_state, symbol_distance):


    V_0_mi_step, V_T_mi_step, atoms_mi_step = NSFR_step(symbol_state, symbol_distance)
    V_0_mi, V_T_mi, atoms_mi = NSFR(V_T_mi_step, atoms_mi_step)


    return V_T_mi, atoms_mi








