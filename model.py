import torch
import numpy as np
from typing import List, Union
from collections import OrderedDict
from abc import ABC, abstractmethod

def create_mlp(
    depth: int,
    in_features: int,
    middle_features: int,
    out_features: int,
    bias: bool = True,
    batchnorm: bool = True,
    final_norm: bool = False,
    final_ac: bool = False
):
    # initial dense layer
    layers = []
    layers.append(
        (
            "linear_1",
            torch.nn.Linear(
                in_features, out_features if depth == 1 else middle_features
            ),
        )
    )
    # iteratively construct batchnorm + relu + dense
    for i in range(depth - 1):
        layers.append(
            (f"batchnorm_{i+1}", torch.nn.BatchNorm1d(num_features=middle_features))
        )
        layers.append((f"relu_{i+1}", torch.nn.ReLU()))
        layers.append(
            (
                f"linear_{i+2}",
                torch.nn.Linear(
                    middle_features,
                    out_features if i == depth - 2 else middle_features,
                    False if i == depth - 2 else bias,
                ),
            )
        )

    if final_norm:
        layers.append(
            (f"batchnorm_{depth}", torch.nn.BatchNorm1d(num_features=out_features))
        )

    if final_ac:
        layers.append(
            (f"final_ac_{depth}", torch.nn.Tanh())
        )

    # return network
    return torch.nn.Sequential(OrderedDict(layers))

class Model(ABC, torch.nn.Module):
    """Abstract model
    Args:
        normalize: whether to normalize after feed-forward
    """
    def __init__(
        self,
        k: int,
        size: int,
        alpha: Union[float, List[float]] = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.k = k
        self.size = size
        self.alpha = alpha
        self.normalize = normalize

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        raise NotImplementedError

    def sample_alpha(self) -> float:
        if isinstance(self.alpha, float) or isinstance(self.alpha, int):
            return self.alpha
        return np.random.uniform(self.alpha[0], self.alpha[1], size=1)[0]

    def post_process(self, dz: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            norm = torch.norm(dz, dim=1)
            dz = dz / torch.reshape(norm, (-1, 1))
        return self.sample_alpha() * dz

class NonlinearConditional(Model):
    """K directions nonlinearly conditional on latent code"""
    def __init__(
        self,
        k: int,
        size: int,
        depth: int,
        alpha: float = 0.1,
        normalize: bool = True,
        bias: bool = True,
        batchnorm: bool = True,
        final_norm: bool = False,
        final_ac: bool = False,
    ) -> None:
        super().__init__(k=k, size=size, alpha=alpha, normalize=normalize)
        self.k = k
        self.size = size

        # make mlp net
        self.nets = torch.nn.ModuleList()

        for i in range(k):
            net = create_mlp(
                depth=depth,
                in_features=size,
                middle_features=size,
                out_features=size,
                bias=bias,
                batchnorm=batchnorm,
                final_norm=final_norm,
                final_ac = final_ac,
            )
            self.nets.append(net)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        #  apply all directions to each batch element
        z = torch.reshape(z, [1, -1, self.size]) # [-1, latent_size] -> [1, -1, latent_size] 
        z = z.repeat(
            (
                self.k,
                1,
                1,
            )
        ) # [-1, latent_size] -> [k, -1, latent_size] 

        #  calculate directions
        dz = []
        for i in range(self.k):
            res_dz = self.nets[i](z[i, ...])
            res_dz = self.post_process(res_dz)
            dz.append(res_dz)

        dz = torch.stack(dz)

        #  add directions
        z = z + dz #[k, -1, latent_size] 

        return torch.reshape(z, [-1, self.size])

    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z + self.post_process(self.nets[k](z))
