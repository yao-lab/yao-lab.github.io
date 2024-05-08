"""Contact map conditioner"""

import torch
from typing import Union
from chroma.layers.structure.conditioners import Conditioner


class ContactMapConditioner(Conditioner):
    """Condition as inter-residue contact map"""
    def __init__(
        self,
        D_inter: torch.Tensor,
        noise_schedule,
        weight: float = 1.0,
        eps: float = 1e-6,
        ca_only:bool = False
    ):
        """
        D_inter: [batch_size, num_residue, num_residue, num_atom_type]
        """
        super().__init__()
        self.D_inter = D_inter
        self.noise_schedule = noise_schedule
        self.weight = weight
        self.eps = eps
        self.ca_only = ca_only
        if ca_only:
            assert D_inter.shape[-1] == 1, f"Using Ca-only inter-residue distance, but has {D_inter.shape[-1]} atom types."

    def _distance(self, X: torch.Tensor):
        dX = X.unsqueeze(2) - X.unsqueeze(1)
        D = torch.sqrt((dX**2).sum(-1) + self.eps)
        return D

    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        O: torch.Tensor,
        U: torch.Tensor,
        t: Union[torch.Tensor, float],
    ):
        if self.ca_only:
            D_inter = self._distance(X[..., 1:2, :])
        else:
            D_inter = self._distance(X)
        loss = (D_inter - self.D_inter).abs()
        scale_t = self.weight * self.noise_schedule.SNR(t).sqrt().clamp(min=1e-3, max=3.0)
        U = U + scale_t * loss
        return X, C, O, U, t