import numpy as np
import torch
import torch.nn as nn
from parna.logger import getLogger


logger = getLogger(__name__)


class TorsionOptimizer(nn.Module):
    def __init__(self, order=4, panelty_weight=0.1, threshold=0.01, fix_phase=True):
        super(TorsionOptimizer, self).__init__()
        self.order = order
        self.k = nn.Parameter(torch.zeros(size=(self.order,)).reshape(1,-1), requires_grad=True)
        self.periodicity = torch.tensor([i + 1 for i in range(order)]).reshape(1,-1)
        phase = torch.zeros((self.order,)).reshape(1,-1)
        for i in range(self.order):
            if i % 2 == 0:
                phase[0,i] = 0
            else:
                phase[0,i] = np.pi
        
        if fix_phase:
            self.phase = phase
        else:
            self.phase = nn.Parameter(phase, requires_grad=True)
        self.fix_phase = fix_phase
        self.mse = nn.MSELoss()
        self.panelty_weight = panelty_weight
        self.threshold = threshold
        logger.info(f"Panelty weight: {self.panelty_weight}")
    
    def forward(self, dihedrals):
        cos_val = (1. + torch.cos( dihedrals * self.periodicity - self.phase )) * self.k
        return cos_val.sum(dim=-1)
    
    def loss(self, energy_pred, energy_true, pairwise=False, weighted=True):
        if weighted:
            weights = self.weight_function(energy_true)   
        else:
            weights = torch.ones_like(energy_true)
        
        if not pairwise:
            energy_pred = energy_pred - energy_pred.min()
            energy_true = energy_true - energy_true.min()
            return torch.sum(weights * (energy_pred - energy_true)**2) / torch.sum(weights) + self.panelty_weight * torch.sum(torch.abs(self.k))
        else:
            pairwise_rel_energy_pred = energy_pred.reshape(-1,1) - energy_pred.reshape(1,-1)
            pairwise_rel_energy_true = energy_true.reshape(-1,1) - energy_true.reshape(1,-1)
            pairwise_weights = (weights.reshape(-1,1) * weights.reshape(1,-1)) ** 0.5  
            loss = torch.sum(pairwise_weights * (pairwise_rel_energy_pred - pairwise_rel_energy_true)**2) / torch.sum(pairwise_weights) / 2.
            return loss + self.panelty_weight * torch.mean(torch.abs(self.k))
        
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
                 n_iter: int = 5000, 
                 lr: float = 1e-3, 
                 weighted: bool = True,
                 pairwise: bool = False):
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
            energy_mm_pred = energy_mm + energy_pred.flatten()
            loss = self.loss(energy_mm_pred, energy_qm, pairwise=pairwise, weighted=weighted)
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                logger.info(f"Iter: {i}, Loss: {loss.item()}")
    
    def infer_parameters(self,
                         dihedrals: np.ndarray,
                         energy_mm: np.ndarray,
                         energy_qm: np.ndarray,
                         pairwise: bool = False):
        dihedral_tensor = torch.from_numpy(dihedrals).float().reshape(-1,1)
        energy_mm_tensor = torch.from_numpy(energy_mm).float().flatten()
        energy_qm_tensor = torch.from_numpy(energy_qm).float().flatten()
        self.optimize(dihedral_tensor, energy_mm_tensor, energy_qm_tensor, n_iter=6501, weighted=True, pairwise=pairwise)
        # drop k if it is less than 1e-2
        if torch.all(torch.abs(self.k) > self.threshold):
            return self.k.detach().numpy()
        else:
            logger.info(self.get_parameters())
            logger.info("Values of k are less than 1e-2. Dropping them and reoptimizing.")
            new_k_set = []
            new_phase = []
            new_periodicity = []
            for i in range(self.order):
                if torch.abs(self.k[0,i]) >= self.threshold:
                    new_k_set.append(self.k[0,i])
                    new_phase.append(self.phase[0,i])
                    new_periodicity.append(self.periodicity[0,i])
            self.k = nn.Parameter(torch.tensor(new_k_set).reshape(1,-1), requires_grad=True)
            if self.fix_phase:
                self.phase = torch.tensor(new_phase).reshape(1,-1)
            else:
                self.phase = nn.Parameter(torch.tensor(new_phase).reshape(1,-1), requires_grad=True)
            self.periodicity = torch.tensor(new_periodicity).reshape(1,-1)
            self.optimize(dihedral_tensor, energy_mm_tensor, energy_qm_tensor, n_iter=3000)
        logger.info(f"FINAL Paremeters: {self.get_parameters()}")
    
    def get_parameters(self):
        if self.fix_phase:
            return {
                "k": self.k.detach().numpy().flatten().tolist(),
                "periodicity": self.periodicity.numpy().flatten().tolist(),
                "phase": self.phase.numpy().flatten().tolist()
            }
        else:
            return {
                "k": self.k.detach().numpy().flatten().tolist(),
                "periodicity": self.periodicity.numpy().flatten().tolist(),
                "phase": self.phase.detach().numpy().flatten().tolist()
            }

