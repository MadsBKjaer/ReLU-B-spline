import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, empty, eye


class BSplineNN(nn.Module):
    __constants__ = ["features", "degree", "n_bplines"]
    features: int
    degree: int
    n_bsplines: int
    knots: Tensor
    weights: Tensor

    def __init__(
        self,
        features: int,
        n_bsplines: int,
        degree: int = 2,
        interval: list[float] = [0, 1],
    ) -> None:
        super().__init__()

        self.features = features
        self.degree = degree
        self.n_bsplines = n_bsplines
        self.interval = interval

        # Select proper ReLU function depending on degree
        if degree == 0:
            self.relu_power = lambda x: torch.heaviside(x, Tensor(1))
        else:
            self.relu_power = lambda x: nn.functional.relu(x).pow(degree)

        self.knots = nn.Parameter(empty((features, n_bsplines + degree + 1)))
        self.weights = nn.Parameter(empty((features, n_bsplines)))
        self.reset_parameter()

    def reset_parameter(self) -> None:
        self.knots = nn.Parameter(
            torch.linspace(
                self.interval[0],
                self.interval[1],
                self.n_bsplines + self.degree + 1,
            ).broadcast_to((self.features, self.n_bsplines + self.degree + 1))
        )

        nn.init.zeros_(self.weights)

    def sort_knots(self) -> None:
        self.knots = nn.Parameter(self.knots.sort()[0])

    def knots_sorted(self) -> bool:
        return (self.knots == self.knots.sort()[0]).all()

    def add_knot(self, index: int) -> None:
        index = np.clip(index, 0, self.n_bsplines + self.degree)

        if index == 0:
            self.knots = nn.Parameter(
                torch.cat(
                    [
                        (self.knots[:, 0] - self.knots[:, :2].diff()).broadcast_to(
                            (self.features, 1)
                        ),
                        self.knots,
                    ],
                    dim=-1,
                )
            )

            self.weights = nn.Parameter(
                torch.cat(
                    [
                        self.weights[:, 0].broadcast_to((self.features, 1)) * 0,
                        self.weights,
                    ],
                    dim=-1,
                )
            )

        elif index == self.n_bsplines + self.degree:
            self.knots = nn.Parameter(
                torch.cat(
                    [
                        self.knots,
                        (self.knots[:, -1] + self.knots[:, -2:].diff()).broadcast_to(
                            (self.features, 1)
                        ),
                    ],
                    dim=-1,
                )
            )

            self.weights = nn.Parameter(
                torch.cat(
                    [
                        self.weights,
                        self.weights[:, -1].broadcast_to((self.features, 1)) * 0,
                    ],
                    dim=-1,
                )
            )

        else:
            self.knots = nn.Parameter(
                torch.cat(
                    [
                        self.knots[:, :index],
                        self.knots[:, index - 1 : index + 1]
                        .mean()
                        .broadcast_to((self.features, 1)),
                        self.knots[:, index:],
                    ],
                    dim=-1,
                )
            )

            self.weights = nn.Parameter(
                torch.cat(
                    [
                        self.weights[:, :index],
                        self.weights[:, index : index + 2]
                        .mean()
                        .broadcast_to((self.features, 1)),
                        self.weights[:, index:],
                    ],
                    dim=-1,
                )
            )

        self.n_bsplines += 1

    def forward(self, input: Tensor) -> Tensor:
        # Forward based on generalized B-spline decomposition
        knots = self.knots.unfold(-1, self.degree + 2, 1)
        relu = self.relu_power(input.unsqueeze(-1).unsqueeze(-1) - knots)
        knot_dif = knots.unsqueeze(-1).mT - knots.unsqueeze(-1) + eye(self.degree + 2)
        coef_matrix = Tensor([1]).div(knot_dif.prod(-1))
        spline_output = torch.linalg.vecdot(relu, coef_matrix)
        output = torch.linalg.vecdot(spline_output, self.weights)
        return output