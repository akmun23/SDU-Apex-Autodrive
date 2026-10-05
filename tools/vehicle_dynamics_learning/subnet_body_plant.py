"""Repository-native, command-independent SUBNET body plant."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class _ResidualNet(nn.Module):
    """deepSI simple_res_net: linear skip plus a tanh MLP residual."""

    def __init__(self, n_in: int, n_out: int, width: int = 64) -> None:
        super().__init__()
        self.net_lin = nn.Linear(n_in, n_out)
        self.net_non_lin = _FeedForward(n_in, n_out, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net_lin(values) + self.net_non_lin(values)


class _FeedForward(nn.Module):
    """Keep deepSI's `net_non_lin.net.*` parameter naming for exact transfer."""

    def __init__(self, n_in: int, n_out: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, n_out),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class SubnetBodyPlant(nn.Module):
    """Predict rear-axle body velocity/yaw from actuator feedback history.

    Inputs are `[steering_feedback_rad, throttle_feedback]`; outputs are
    `[u_rear_mps, v_rear_mps, yaw_rate_rps]`. `step` returns the observation
    associated with the current latent state, then advances that state using
    the supplied actuator feedback, matching the frozen deepSI SUBNET timing.
    """

    def __init__(self, input_mean, input_std, output_mean, output_std,
                 history_steps: int = 12, state_order: int = 9,
                 width: int = 64) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.state_order = int(state_order)
        self.encoder = _ResidualNet(
            self.history_steps * (2 + 3), self.state_order, width)
        self.fn = _ResidualNet(self.state_order + 2, self.state_order, width)
        self.hn = _ResidualNet(self.state_order, 3, width)
        self.register_buffer("input_mean", torch.as_tensor(input_mean, dtype=torch.float32))
        self.register_buffer("input_std", torch.as_tensor(input_std, dtype=torch.float32))
        self.register_buffer("output_mean", torch.as_tensor(output_mean, dtype=torch.float32))
        self.register_buffer("output_std", torch.as_tensor(output_std, dtype=torch.float32))

    def encode_history(self, u_past: torch.Tensor,
                       y_past: torch.Tensor) -> torch.Tensor:
        """Encode measured actuator/body history ending at the current sample."""
        single = u_past.ndim == 2
        if single:
            u_past, y_past = u_past.unsqueeze(0), y_past.unsqueeze(0)
        if (u_past.ndim != 3 or y_past.ndim != 3
                or u_past.shape[1:] != (self.history_steps, 2)
                or y_past.shape[1:] != (self.history_steps, 3)
                or u_past.shape[0] != y_past.shape[0]):
            raise ValueError("history must have aligned [batch, 12, 2/3] shapes")
        u_norm = (u_past - self.input_mean) / self.input_std
        y_norm = (y_past - self.output_mean) / self.output_std
        encoded = self.encoder(torch.cat((u_norm.flatten(1), y_norm.flatten(1)), dim=1))
        return encoded[0] if single else encoded

    def step(self, z: torch.Tensor,
             actuator_feedback: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current-state body output and next latent state."""
        single = z.ndim == 1
        if single:
            z = z.unsqueeze(0)
            actuator_feedback = actuator_feedback.unsqueeze(0)
        if z.ndim != 2 or z.shape[-1] != self.state_order:
            raise ValueError("latent state must have shape [batch, state_order]")
        if actuator_feedback.shape != (z.shape[0], 2):
            raise ValueError("actuator feedback must have shape [batch, 2]")
        u_norm = (actuator_feedback - self.input_mean) / self.input_std
        y = self.hn(z) * self.output_std + self.output_mean
        next_z = self.fn(torch.cat((z, u_norm), dim=1))
        return (y[0], next_z[0]) if single else (y, next_z)

    def rollout(self, z0: torch.Tensor,
                future_inputs: torch.Tensor) -> torch.Tensor:
        """Recursively predict body outputs from future actuator feedback."""
        single = z0.ndim == 1
        if single:
            z0 = z0.unsqueeze(0)
            future_inputs = future_inputs.unsqueeze(0)
        if (z0.ndim != 2 or z0.shape[-1] != self.state_order
                or future_inputs.ndim != 3
                or future_inputs.shape[0] != z0.shape[0]
                or future_inputs.shape[-1] != 2):
            raise ValueError("rollout expects [batch, state] and [batch, time, 2]")
        state = z0
        predictions = []
        for index in range(future_inputs.shape[1]):
            output, state = self.step(state, future_inputs[:, index])
            predictions.append(output)
        result = torch.stack(predictions, dim=1)
        return result[0] if single else result

    def save_npz(self, path: str | Path) -> None:
        """Write portable numeric weights; no serialized Python modules."""
        arrays = {
            "format_version": np.asarray([1], dtype=np.int32),
            "history_steps": np.asarray([self.history_steps], dtype=np.int32),
            "state_order": np.asarray([self.state_order], dtype=np.int32),
            "input_mean": self.input_mean.detach().cpu().numpy(),
            "input_std": self.input_std.detach().cpu().numpy(),
            "output_mean": self.output_mean.detach().cpu().numpy(),
            "output_std": self.output_std.detach().cpu().numpy(),
        }
        arrays.update({
            f"weight__{name}": value.detach().cpu().numpy()
            for name, value in self.state_dict().items()
            if name not in {"input_mean", "input_std", "output_mean", "output_std"}
        })
        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path: str | Path) -> "SubnetBodyPlant":
        with np.load(path, allow_pickle=False) as source:
            if int(source["format_version"][0]) != 1:
                raise ValueError("unsupported SUBNET checkpoint format")
            model = cls(
                source["input_mean"], source["input_std"],
                source["output_mean"], source["output_std"],
                history_steps=int(source["history_steps"][0]),
                state_order=int(source["state_order"][0]))
            state = model.state_dict()
            for name in state:
                key = f"weight__{name}"
                if key in source:
                    state[name] = torch.as_tensor(source[key], dtype=state[name].dtype)
            model.load_state_dict(state, strict=True)
        return model.eval()
