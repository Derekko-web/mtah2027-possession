"""Fine-tune an external temporal network with a shared player-set encoder."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dfl_submission import motion_cpu, motion_features, temporal_cpu, temporal_network
from dfl_submission.train import write_json
from experiments.dfl_transfer_data import digest


BATCH = 12
CORE = 96


class MotionNetwork(nn.Module):
    def __init__(self, columns):
        super().__init__()
        self.base = temporal_network.Network(columns)
        self.embed1 = nn.Linear(motion_features.CHANNELS, motion_cpu.EMBED)
        self.embed2 = nn.Linear(motion_cpu.EMBED, motion_cpu.EMBED)
        self.motion = nn.Linear(4 * motion_cpu.EMBED, temporal_cpu.WIDTH)
        nn.init.zeros_(self.motion.weight)
        nn.init.zeros_(self.motion.bias)

    def forward(self, values, players):
        visible = players[..., -1:] > 0
        if self.training:
            visible = visible & (torch.rand_like(players[..., -1:]) >= .06)
        encoded = F.gelu(self.embed2(F.gelu(self.embed1(players))))
        count = visible.sum(dim=-2)
        average = torch.where(visible, encoded, 0).sum(dim=-2) / count.clamp_min(1)
        maximum = torch.where(visible, encoded, -1000.).amax(dim=-2)
        pooled = torch.cat((average, torch.where(count > 0, maximum, 0)), dim=-1).flatten(start_dim=2)
        hidden = self.base.project(values.transpose(1, 2)) + self.motion(pooled).transpose(1, 2)
        hidden = self.base.temporal(hidden).transpose(1, 2)
        return self.base.outputs(self.base.norm(hidden))


def load_training(context_folder, motion_folder):
    grouped, base, permutation, location, scale = temporal_network.training_data(context_folder)
    motion = json.loads((motion_folder / "manifest.json").read_text())
    if motion["base_manifest"] != digest(context_folder / "manifest.json"):
        raise ValueError("Player motion and context training sets differ")
    sets = defaultdict(list)
    for entry in motion["records"]:
        path = motion_folder / entry["file"]
        if digest(path) != entry["player_sha256"]:
            raise ValueError(f"Player-motion training input differs: {path}")
        with np.load(path, allow_pickle=False) as array:
            players = array["players"]
        index = len(sets[entry["source"]])
        context, labels = grouped[entry["source"]][index]
        if len(context) != len(players):
            raise ValueError("Player-motion time axis differs")
        sets[entry["source"]].append((context, players, labels))
    return sets, base, permutation, location, scale


def batch(groups, rng, permutation, location, scale):
    halo = temporal_network.HALO
    length = CORE + 2 * halo
    context, motions, targets = [], [], []
    for source in sorted(groups):
        for _ in range(BATCH // len(groups)):
            rows, players, labels = groups[source][rng.integers(len(groups[source]))]
            start = rng.integers(0, len(rows) - length + 1)
            values, slots = rows[start:start + length], players[start:start + length]
            output = labels[start + halo:start + halo + CORE]
            if rng.integers(2):
                values, slots = values[:, permutation], slots[:, ::-1]
                output = output.copy()
                output[:, 1] = 1 - output[:, 1]
            context.append(temporal_cpu.normalize(values, location, scale))
            motions.append(slots)
            targets.append(output)
    return (torch.from_numpy(np.stack(context)), torch.from_numpy(np.stack(motions).astype(np.float32)),
            torch.from_numpy(np.stack(targets)))


def train(context, motion, previous, output, device):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    torch.manual_seed(73)
    rng = np.random.default_rng(73)
    groups, manifest, permutation, location, scale = load_training(context, motion)
    base = json.loads((previous / "fit.json").read_text())
    checkpoint_name = next(name for name in ("temporal_1800.pt", "temporal_1800.npz") if name in base["models"])
    source = previous / checkpoint_name
    norm = previous / "normalization.npz"
    if (base["train_manifest"] != digest(context / "manifest.json") or base["columns"] != manifest["columns"]
            or digest(source) != base["models"][source.name] or digest(norm) != base["normalization"]):
        raise ValueError("Warm start does not match the verified external training inputs")
    with np.load(norm, allow_pickle=False) as prior_norm:
        for name, actual in (("location", location), ("scale", scale), ("permutation", permutation)):
            np.testing.assert_array_equal(prior_norm[name], actual)
    copyfile(norm, output / norm.name)
    net = MotionNetwork(len(location))
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as arrays:
            state = {key: torch.from_numpy(arrays[key]) for key in arrays.files}
    else:
        state = torch.load(source, map_location="cpu", weights_only=True)
    net.base.load_state_dict(state)
    net.to(device)
    optimizer = torch.optim.AdamW([{"params": net.base.parameters(), "lr": .00028},
                                  {"params": [p for name, p in net.named_parameters() if not name.startswith("base.")],
                                   "lr": .00085}], weight_decay=.012)
    hashes = {}
    halo = temporal_network.HALO
    for step in range(1, motion_cpu.STEPS[-1] + 1):
        values, players, labels = (part.to(device) for part in batch(groups, rng, permutation, location, scale))
        multiplier = .25 + .75 * .5 * (1 + math.cos(math.pi * (step - 1) / motion_cpu.STEPS[-1]))
        for group, lr in zip(optimizer.param_groups, (.00028, .00085)):
            group["lr"] = lr * multiplier
        optimizer.zero_grad(set_to_none=True)
        loss = temporal_network.loss(net(values, players)[:, halo:halo + CORE], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"Player-motion training step {step}, loss {loss.item():.4f}", flush=True)
        if step in motion_cpu.STEPS:
            path = output / f"motion_{step}.npz"
            np.savez(path, **{key: value.detach().cpu().numpy() for key, value in net.state_dict().items()})
            hashes[path.name] = digest(path)
    write_json(output / "fit.json", {"models": hashes, "normalization": digest(output / norm.name),
               "columns": manifest["columns"], "training_device": device, "torch": torch.__version__,
               "motion_manifest": digest(motion / "manifest.json"), "warm_start": digest(source)})


def verify(folder):
    torch.set_num_threads(1)
    fit = json.loads((folder / "fit.json").read_text())
    net = MotionNetwork(len(fit["columns"])).eval()
    rng = np.random.default_rng(17)
    values = rng.normal(size=(2, 185, len(fit["columns"]) * 2)).astype(np.float32)
    players = rng.normal(size=(2, 185, 2, motion_features.PLAYERS, motion_features.CHANNELS)).astype(np.float32)
    players[..., -1] = rng.random(players.shape[:-1]) > .3
    differences = {}
    for name in fit["models"]:
        weights, *_ = motion_cpu.load(folder, name)
        net.load_state_dict({key: torch.from_numpy(array) for key, array in weights.items()})
        with torch.inference_mode():
            expected = net(torch.from_numpy(values), torch.from_numpy(players)).numpy()
        actual = motion_cpu.forward(values, players, weights)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)
        differences[name] = float(np.max(np.abs(actual - expected)))
    write_json(folder / "export_validation.json", differences)
    print(f"Player-motion CPU export verified: {differences}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    fitting = commands.add_parser("train")
    fitting.add_argument("--context", required=True, type=Path)
    fitting.add_argument("--motion", required=True, type=Path)
    fitting.add_argument("--previous", required=True, type=Path)
    fitting.add_argument("--out", required=True, type=Path)
    fitting.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    check = commands.add_parser("verify")
    check.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "train":
        train(args.context, args.motion, args.previous, args.out, args.device)
    else:
        verify(args.model)
