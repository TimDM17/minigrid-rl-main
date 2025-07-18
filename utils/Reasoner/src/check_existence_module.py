import torch
import torch.nn as nn
import torch.nn.functional as F
from .fol.logic import PlannerPredicate
import tqdm
from .fol.logic import NeuralPredicate, Predicate

import numpy as np

class Check_Existence_module(nn.Module):
    """
    FactsConverter converts the output from the perception module to the valuation vector.
    """
    def __init__(self, lang, clause, device=None):
        super(Check_Existence_module, self).__init__()
        #self.e = perception_module.e
        #self.d = perception_module.d
        self.lang = lang
        self.vm = None  # valuation functions
        self.device = device
        self.object_clause = clause

    def __str__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def __repr__(self):
        return "FactsConverter(entities={}, dimension={})".format(self.e, self.d)

    def forward(self, Z,G,V):
        return self.convert(Z,G,V)

    def get_params(self):
        return self.vm.get_params()

    def init_valuation(self, n, batch_size):
        v = torch.zeros((batch_size, n)).to(self.device)
        v[:, 1] = 1.0
        return v

    def filter_by_datatype(self):
        pass

    def to_vec(self, term, zs):
        pass

    def __convert(self, Z, G):
        # Z: batched output
        vs = []
        for zs in tqdm(Z):
            vs.append(self.convert_i(zs, G))
        return torch.stack(vs)


    def tranform_object_clause_into_move_atoms(self):
        # Z: batched output
        a = []
        for clause in self.object_clause:
            b = str('move') + '(' + str(clause.body[1]) + ',' + str(clause.body[0]) + ',' + str(
                clause.head) + ')'
            a.append(b)
        return a


    def convert(self, object_atoms, G, symbol_state):
        'symbol_state is used to represent the observed state of the agent'
        'blue key, red key, blue door closed, blue door open, red door closed, red door open, reach goal'
        'blue key, red key, blue door, red door, goal'

        batch_size = 1
        true_move_atoms = self.tranform_object_clause_into_move_atoms()
        V = torch.zeros((batch_size, len(G))).to(torch.float32).to(self.device)

        if len(symbol_state) == 7:
            symbol_state = [symbol_state[0], symbol_state[1],  symbol_state[3],  symbol_state[5], symbol_state[6]]


        # Create a new tensor for updating V without in-place operations

        for i, atom in enumerate(G):
            if atom.pred.name == 'move':
                'assign the move based on the symbol state'
                V[:, 3] = symbol_state[1]
               # V[:, 3] = 1
                V[:, 4] = symbol_state[2]
                #V[:, 4] = 0
                V[:, 5] = symbol_state[4]
                #V[:, 5] = max(symbol_state[1], symbol_state[2]) if len(symbol_state) == 7 else symbol_state[-1]
               # V[:, 5] = 1
                V[:, 6] = symbol_state[3]
                #V[:, 6] = symbol_state[1] if len(symbol_state) == 7 else symbol_state[3]
               # V[:, 6] = 1
            if atom.pred.name == 'equal':
                V[:, i] = 1
            if atom.pred.name == 'plan':
                if str(atom) == 'plan(initial(A),initial(A),rg(A,G),*)':
                    V[:, i] = 1
                if str(atom) == 'plan(initial(A),bldo(A,B),rg(A,G),*)':
                    V[:, i] = 1

        V[:, 1] = 1



        return V

    def check_value(self,Z, Z_pred, atom ):
        index = np.where(Z_pred == str(atom.pred))

        return Z[index]




    def convert_i(self, zs, G):
        v = self.init_valuation(len(G))
        for i, atom in enumerate(G):
            if type(atom.pred) == PlannerPredicate and i > 1:
                v[i] = self.vm.eval(atom, zs)
        return v

    def call(self, pred):
        return pred

class SquareClassifier(nn.Module):
    def __init__(self):
        super(SquareClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Convolutional Layer 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Convolutional Layer 2
        self.pool = nn.MaxPool2d(2, 2)  # Pooling Layer
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Fully Connected Layer 1
        self.fc2 = nn.Linear(128, 1)  # Fully Connected Layer 2 (binary classification)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Convolution -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Convolution -> ReLU -> Pooling
        x = x.view(-1, 32 * 7 * 7)  # Flatten the output for fully connected layers
        x = F.relu(self.fc1(x))  # Fully Connected -> ReLU
        x = torch.sigmoid(self.fc2(x))  # Output layer -> Sigmoid (binary classification)
        return x