import numpy as np
import torch
import torch.nn as nn
from parna.logger import getLogger
import random
from pathlib import Path


logger = getLogger(__name__)


class TorsionOptimizer(nn.Module):
    def __init__(self, order=4, panelty_weight=0.1, threshold=0.01, fix_phase=True, n_dihedrals=1, seed=1106):
        super(TorsionOptimizer, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        self.order = order
        self.n_dihedrals = n_dihedrals
        self.k = nn.Parameter(torch.zeros(size=(self.order*n_dihedrals,)).reshape(1, n_dihedrals, self.order), requires_grad=True)
        self.periodicity = torch.zeros((self.order*n_dihedrals,)).reshape(1, n_dihedrals, self.order)
        phase = torch.zeros((self.order*n_dihedrals,)).reshape(1, n_dihedrals, self.order)
        for i in range(self.order):
            self.periodicity[0,:,i] = i + 1
            if i % 2 == 0:
                phase[0,:,i] = 0
            else:
                phase[0,:,i] = np.pi
        
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
        return cos_val.sum(dim=(-1, -2))
    
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
        energy_mm_tensor = torch.from_numpy(energy_mm).float().flatten()
        energy_qm_tensor = torch.from_numpy(energy_qm).float().flatten()
        # of shape (n_conformers, n_dihedrals, orders)
        dihedral_tensor = torch.from_numpy(dihedrals).float().reshape(-1, self.n_dihedrals, 1)
        
        self.optimize(dihedral_tensor, energy_mm_tensor, energy_qm_tensor, n_iter=8501, weighted=True, pairwise=pairwise)
        # drop k if it is less than 1e-2
        # if torch.all(torch.abs(self.k) > self.threshold):
        #     return self.k.detach().numpy()
        # else:
        #     logger.info(self.get_parameters())
        #     logger.info("Values of k are less than 1e-2. Dropping them and reoptimizing.")
        #     new_k_set = []
        #     new_phase = []
        #     new_periodicity = []
        #     for i in range(self.order):
        #         if torch.abs(self.k[0,i]) >= self.threshold:
        #             new_k_set.append(self.k[0,i])
        #             new_phase.append(self.phase[0,i])
        #             new_periodicity.append(self.periodicity[0,i])
        #     self.k = nn.Parameter(torch.tensor(new_k_set).reshape(1,-1), requires_grad=True)
        #     if self.fix_phase:
        #         self.phase = torch.tensor(new_phase).reshape(1,-1)
        #     else:
        #         self.phase = nn.Parameter(torch.tensor(new_phase).reshape(1,-1), requires_grad=True)
        #     self.periodicity = torch.tensor(new_periodicity).reshape(1,-1)
        #     self.optimize(dihedral_tensor, energy_mm_tensor, energy_qm_tensor, n_iter=5000)
        logger.info(f"FINAL Paremeters: {self.get_parameters()}")
    
    def get_parameters(self):
        if self.fix_phase:
            return {
                "k": self.k.reshape(self.n_dihedrals, self.order).detach().numpy().tolist(),
                "periodicity": self.periodicity.reshape(self.n_dihedrals, self.order).numpy().tolist(),
                "phase": self.phase.reshape(self.n_dihedrals, self.order).numpy().tolist()
            }
        else:
            return {
                "k": self.k.reshape(self.n_dihedrals, self.order).detach().numpy().tolist(),
                "periodicity": self.periodicity.reshape(self.n_dihedrals, self.order).numpy().tolist(),
                "phase": self.phase.reshape(self.n_dihedrals, self.order).detach().numpy().tolist()
            }


def shift_angle_periodicity(angle):
    """Shift angle to the range of [0, 2*pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_idx(angle, base_value=-180., inverval=15., atol=0.08    ):
    angle_idx = (angle - base_value) / inverval
    angle_coeff = {}
    if np.isclose(angle_idx, int(angle_idx), atol=atol):
        angle_coeff[int(angle_idx)] = 1.
    elif np.isclose(angle_idx, int(angle_idx)+1, atol=atol):
        angle_coeff[int(angle_idx)+1] = 1.
    else:
        angle_coeff[int(angle_idx)+1] = angle_idx - int(angle_idx)
        angle_coeff[int(angle_idx)] = 1 - angle_coeff[int(angle_idx)+1]
    return angle_coeff
    

def get_idx_from_angle(sugar, epsilon, zeta, fmt="drgree"):
    if fmt == "rad":
        sugar = np.rad2deg(sugar)
        epsilon = np.rad2deg(epsilon)
        zeta = np.rad2deg(zeta)
    
    # print(sugar, epsilon, zeta)
    epsilon_coeff = get_idx(epsilon, base_value=-180., inverval=15.)
    zeta_coeff = get_idx(zeta, base_value=-180., inverval=15.)
    
    all_coeff = {}
    for epi, ec in epsilon_coeff.items():
        for zetai, zc in zeta_coeff.items():
            all_coeff[(epi%24)*24+(zetai%24)] = ec * zc

    sugar_coeff = get_idx(sugar, base_value=-60., inverval=15.)
    for sugari, sc in sugar_coeff.items():
        for epi, ec in epsilon_coeff.items():
            if epi == 24: epi = 0
            if sugari == 24: sugari = 0
            assert epi < 24, f"epi: {epi} sugari: {sugari}, sugar: {sugar}, epsilon: {epsilon}, zeta: {zeta}"
            assert sugari < 24, f"epi: {epi} sugari: {sugari}, sugar: {sugar}, epsilon: {epsilon}, zeta: {zeta}"
            all_coeff[(sugari%24)*24+(epi%24)+24*24] = ec * sc
    return all_coeff


class LinearCMAPSolver:
    def __init__(self, n_param, ):     
        self.n_param = n_param
    
    def fit(self, all_energy_info, output_dir):
        n_keys = len(list(all_energy_info.keys()))
        assert n_keys >= self.n_param
        coeff_matrix = np.zeros((n_keys,self.n_param))

        nnp_energy_vector = np.zeros(n_keys)
        mm_energy_vector = np.zeros(n_keys)
        for i, conf in enumerate(list(all_energy_info.keys())):
            info = all_energy_info[conf]
            dihedral_angle = info["dihedral"]
            all_coeff = get_idx_from_angle(shift_angle_periodicity(dihedral_angle["sugar"]), 
                                           shift_angle_periodicity(dihedral_angle["epsilon"]), 
                                           shift_angle_periodicity(dihedral_angle["zeta"]),
                                           fmt="rad")
            for coeff_i, coeff in all_coeff.items():
                coeff_matrix[i,coeff_i] = coeff
            nnp_energy_vector[i] = info["qm_energy"]
            mm_energy_vector[i] = info["mm_energy"]
        
        nnp_energy_vector = nnp_energy_vector - nnp_energy_vector.min()
        mm_energy_vector = mm_energy_vector - mm_energy_vector.min()
        cmap_energy = nnp_energy_vector - mm_energy_vector
        x, residuals, rank, s = np.linalg.lstsq(coeff_matrix, cmap_energy, rcond=None)
        epsilon_zeta = x[:24*24].reshape(24,24)
        sugar_epsilon = x[24*24:].reshape(-1, 24)
        head_patch = np.zeros((8,24))
        tail_patch = np.zeros((7,24))

        for i in range(8):
            head_patch[7-i] = sugar_epsilon[0] * np.exp(-(i+1)/2)

        for i in range(7):
            tail_patch[i] = sugar_epsilon[-1] * np.exp(-(i+1)/2)

        sugar_epsilon_cmap = np.concatenate([head_patch, sugar_epsilon, tail_patch])
        epsilon_zeta_cmap = epsilon_zeta
        output_dir = Path(output_dir)
        np.save(output_dir / "sugar_epsilon_cmap.npy", sugar_epsilon_cmap)
        np.save(output_dir / "epsilon_zeta_cmap.npy", epsilon_zeta_cmap)

        
   