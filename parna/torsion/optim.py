import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from parna.logger import getLogger
from .module import TorsionFragmentizer

from pathlib import Path
import rdkit.Chem as Chem

from parna.utils import rd_load_file


logger = getLogger(__name__)


class TorsionOptimizer(nn.Module):
    def __init__(self, order=4, panelty_weight=0.1, threshold=0.01):
        super(TorsionOptimizer, self).__init__()
        self.order = order
        self.k = nn.Parameter(torch.zeros(size=(self.order,)), requires_grad=True)
        self.periodicity = torch.tensor([i + 1 for i in range(order)])
        self.phase = torch.zeros((self.order,))
        for i in range(self.order):
            if i % 2 == 0:
                self.phase[i] = 0
            else:
                self.phase[i] = np.pi
        
        self.mse = nn.MSELoss()
        self.panelty_weight = panelty_weight
        self.threshold = threshold
        logger.info(f"Panelty weight: {self.panelty_weight}")
    
    def forward(self, dihedrals):
        cos_val = torch.cos( dihedrals * self.periodicity + self.phase ) * self.k
        return cos_val.sum(dim=-1)
    
    def loss_weighted(self, energy_pred, energy_true):
        weights = self.weight_function(energy_true)      
        energy_pred = energy_pred - energy_pred.min()
        energy_true = energy_true - energy_true.min()
        return torch.sum(weights * (energy_pred - energy_true)**2) / torch.sum(weights) + self.panelty_weight * torch.sum(torch.abs(self.k))
    
    def loss(self, energy_pred, energy_true):
        energy_pred = energy_pred - energy_pred.min()
        energy_true = energy_true - energy_true.min()
        return torch.sum((energy_pred - energy_true)**2) / energy_pred.size(0) + self.panelty_weight * torch.sum(torch.abs(self.k))
        
    def weight_function(self, energy, lower_bound=1., upper_bound=10.):
        """Weight function to assign different weights to different energy values.
            1 if energy < 1 kcal/mol
            (1+(E-1)^2)^-(1/2) if 1 < energy < 10 kcal/mol
            0 if energy > 10 kcal/mol
        """
        weights = torch.where(energy < lower_bound, 1., 0.)
        masks = (energy > lower_bound) & (energy < upper_bound)
        weights[masks] = 1. / torch.sqrt(1 + (energy[masks] - 1)**2)
        return  weights
    
    def optimize(self, 
                 dihedrals: torch.Tensor,
                 energy_mm: torch.Tensor, 
                 energy_qm: torch.Tensor, 
                 n_iter: int = 1000, 
                 lr: float = 1e-3, 
                 weighted: bool = True):
        """Optimize the torsion angles. All inputs are torch.Tensor.
        
        Input:
            dihedrals: torch.Tensor, shape (n_conformers, n_dihedrals)
            energy_mm: torch.Tensor, shape (n_conformers, ). The MM energy without torsion term. E_MM - E_MM_torsion
            energy_qm: torch.Tensor, shape (n_conformers, ). The QM energy.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        logger.info(f"Start optimization for model with periodicity of {self.order}")
        for i in range(n_iter):
            optimizer.zero_grad()
            energy_pred = self(dihedrals)
            energy_mm_pred = energy_mm+energy_pred.flatten()
            if weighted:
                loss = self.loss_weighted(energy_mm_pred, energy_qm)
            else:
                loss = self.loss(energy_pred, energy_qm)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logger.info(f"Iter: {i}, Loss: {loss.item()}")
        return self.k.detach().numpy()
    
    def infer_parameters(self,
                         dihedrals: np.ndarray,
                         energy_mm: np.ndarray,
                         energy_qm: np.ndarray,
        ):
        dihedral_tensor = torch.from_numpy(dihedrals).float()
        energy_mm_tensor = torch.from_numpy(energy_mm).float()
        energy_qm_tensor = torch.from_numpy(energy_qm).float()
        self.optimize(dihedral_tensor, energy_mm_tensor, energy_qm_tensor)
        # drop k if it is less than 1e-2
        if torch.all(self.k > self.threshold):
            return self.k.detach().numpy()
        else:
            print("Values of k are less than 1e-2. Dropping them and reoptimizing.")
        new_k_set = []
        new_phase = []
        new_periodicity = []
        for i in range(self.order):
            if self.k[i] >= self.threshold:
                new_k_set.append(self.k[i])
                new_phase.append(self.phase[i])
                new_periodicity.append(self.periodicity[i])
        self.k = nn.Parameter(torch.tensor(new_k_set), requires_grad=True)
        self.phase = torch.tensor(new_phase)
        self.periodicity = torch.tensor(new_periodicity)
        self.optimize(dihedral_tensor, energy_mm_tensor, energy_qm_tensor)
    
    def get_parameters(self):
        return {
            "k": self.k.detach().numpy(),
            "periodicity": self.periodicity.numpy(),
            "phase": self.phase.numpy()
        }

