import torch
import torch.nn as nn
from torch import Tensor, empty, eye


def pprint(tensor, name: str):
    print(f"\n{name}: {tensor.shape}\n{tensor}")


class BSpline_v1(nn.Module):
    __constants__   = ["in_features", "out_features", "degree"]
    in_features: int
    out_features: int
    degree: int
    knots: Tensor

    def __init__(self, in_features: int, out_features: int, degree: int, share_knots: bool) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.share_knots = share_knots
        self.relu_power = lambda x: nn.functional.relu(x).pow(degree)
        self.linear = nn.Linear(in_features, out_features, bias = False)

        if share_knots:
            self.knots = nn.Parameter(empty((in_features + degree + 1)))
        else:
            self.knots = nn.Parameter(empty((in_features, degree + 2)))

        self.reset_parameter()

    def reset_parameter(self) -> None:
        nn.init.uniform_(self.knots)

    def forward(self, input: Tensor) -> Tensor:
        if self.share_knots:
            knots = self.knots.unfold(0, self.degree + 2, 1)
        else:
            knots = self.knots

        shifted_relu    = self.relu_power(input.unsqueeze(-1) - knots).pow(self.degree)
        knot_dif        = knots.unsqueeze(1).mT - knots.unsqueeze(1) + eye(self.degree + 2)
        coef_matrix     = Tensor([1]).div(knot_dif.prod(1))
        spline_output   = torch.linalg.vecdot(shifted_relu, coef_matrix)
        return self.linear(spline_output)


class BSpline_v2(nn.Module):
    __constants__ = ["features", "degree", "n_bplines", "share_knots", "epsilon"]
    features: int
    degree: int
    n_bsplines: int
    knots: Tensor
    weights: Tensor
    epsilon: float

    def __init__(self, features: int, n_bsplines: int, degree: int = 2, share_knots: bool = True, interval: list[float] = [0, 1], epsilon: float = 1e-16) -> None:
        super().__init__()

        self.features = features
        self.degree = degree
        self.n_bsplines = n_bsplines
        self.share_knots = share_knots
        self.interval = interval
        self.epsilon = epsilon

        if degree == 0:
            self.relu_power = lambda x: torch.heaviside(x, Tensor(1))
        else:
            self.relu_power = lambda x: nn.functional.relu(x).pow(degree)

        if share_knots:
            self.knots = nn.Parameter(empty((features, n_bsplines + degree + 1)))
        else:
            self.knots = nn.Parameter(empty((features, n_bsplines, degree + 2)))
        self.weight = nn.Parameter(empty((features, n_bsplines)))
        self.reset_parameter()

    def reset_parameter(self) -> None:
        if self.share_knots:
            self.knots = nn.Parameter(torch.linspace(self.interval[0], self.interval[1], self.n_bsplines + self.degree + 1).broadcast_to((self.features, self.n_bsplines + self.degree + 1)) + nn.init.normal_(self.knots))
        else:
            self.knots = nn.Parameter(torch.linspace(self.interval[0], self.interval[1], self.n_bsplines + self.degree + 1).unfold(-1, self.degree + 2, 1).broadcast_to((self.features, self.n_bsplines, self.degree + 2)) + nn.init.normal_(self.knots))

        nn.init.normal_(self.weight)

        # print(self.weights)
        # print(self.knots)

    def forward(self, input: Tensor) -> Tensor:
        if self.share_knots:
            knots = self.knots.unfold(-1, self.degree + 2, 1)
        else:
            knots = self.knots

        # print(knots)

        relu = self.relu_power(input.unsqueeze(-1).unsqueeze(-1) - knots)
        # print(relu[0, 0])
        knot_dif = knots.unsqueeze(-1).mT - knots.unsqueeze(-1) + eye(self.degree + 2)
        # print(knot_dif[0, 0])
        coef_matrix     = Tensor([1]).div((knot_dif + self.epsilon).prod(-1))
        # print(coef_matrix[0])
        spline_output   = torch.linalg.vecdot(relu, coef_matrix)
        # print(spline_output[0, 0])
        output = torch.linalg.vecdot(spline_output, self.weight)
        # print(output[0, 0])
        return output


class ClosedForm(nn.Module):
    __constants__ = ["features"]
    features: int

    def __init__(self, features: int) -> None:
        super().__init__()

        self.sinus_weight = nn.Parameter(empty(features, 1))
        self.sinus_bias = nn.Parameter(empty(features, 1))
        self.power_weight = nn.Parameter(empty(features, 1))
        self.power_bias = nn.Parameter(empty(features, 1))
        self.linear_weight = nn.Parameter(empty(features, 1))
        self.linear_bias = nn.Parameter(empty(features, 1))

        self.reset_parameter()

    def reset_parameter(self) -> None:
        nn.init.normal_(self.sinus_weight)
        nn.init.normal_(self.sinus_bias)
        nn.init.normal_(self.power_weight)
        nn.init.normal_(self.power_bias)
        nn.init.normal_(self.linear_weight)
        nn.init.normal_(self.linear_bias)

    def forward(self, input: Tensor) -> Tensor:
        sinus = torch.sin(input) @ self.sinus_weight + self.sinus_bias
        power = input.pow(2) @ self.power_weight + self.power_bias
        linear = input @ self.linear_weight + self.linear_bias
        return sinus + power + linear