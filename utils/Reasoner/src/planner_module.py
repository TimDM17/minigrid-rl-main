import torch
import torch.nn as nn
from .fol.logic import PlannerPredicate



class Planner_module(nn.Module):
    """
    FactsConverter converts the output from the perception module to the valuation vector.
    """

    def __init__(self, lang, perception_module, valuation_module, device=None):
        super(Planner_module, self).__init__()
        self.e = perception_module.e
        self.d = perception_module.d
        self.lang = lang
        self.vm = valuation_module  # valuation functions
        self.device = device

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

    def filter_by_datatype():
        pass

    def to_vec(self, term, zs):
        pass

    def __convert(self, Z, G):
        # Z: batched output
        vs = []
        for zs in tqdm(Z):
            vs.append(self.convert_i(zs, G))
        return torch.stack(vs)

    def convert(self, Z, G, V):
        batch_size = Z.size(0)

        # V = self.init_valuation(len(G), Z.size(0))
     #   V = torch.zeros((batch_size, len(G))).to(
     #       torch.float32).to(self.device)
        for i, atom in enumerate(G):
            if type(atom.pred) == PlannerPredicate and i > 1:
                V[:, i] = self.vm(Z, atom)
         #   elif atom in B:
         #
         #       V[:, i] += torch.ones((batch_size, )).to(
         #           torch.float32).to(self.device)
   #     V[:, 1] = torch.ones((batch_size, )).to(
   #         torch.float32).to(self.device)

      #  move_step = self.move_step(V)

        return V#, move_step

    def move_step(self, V):
        direction = torch.reshape(V[0][37:53], (4,4)) *torch.FloatTensor([-1,-1,1,1])
        step_hori = torch.reshape(V[0][101:121], (4,5)) *torch.FloatTensor([0,1,2,3,4])
        step_ver = torch.reshape(V[0][121::], (4,5)) *torch.FloatTensor([0,1,2,3,4])

        move_hori = (direction[:,0]*torch.reshape(step_hori.sum(dim=1),(1,4)) + direction[:,1]*torch.reshape(step_hori.sum(dim=1),(1,4))).sum(dim=0)
        move_ver = (direction[:,2]*torch.reshape(step_ver.sum(dim=1),(1,4)) + direction[:,3]*torch.reshape(step_ver.sum(dim=1),(1,4))).sum(dim=0)

        move_step = torch.stack((move_hori,move_ver))

        return move_step

    def convert_i(self, zs, G):
        v = self.init_valuation(len(G))
        for i, atom in enumerate(G):
            if type(atom.pred) == PlannerPredicate and i > 1:
                v[i] = self.vm.eval(atom, zs)
        return v

    def call(self, pred):
        return pred
