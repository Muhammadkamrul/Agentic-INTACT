"""
intact_agentic/agent/nn.py
==========================
A small hand-written neural-network toolkit in pure NumPy.

WHY NOT PYTORCH
---------------
Three reasons, in order of importance.

1. **Deployment realism.**  The thing being modelled is an xApp inside a
   near-RT RIC with a 10 ms to 1 s budget.  A forward pass through a
   4-layer, 32-unit network on a handful of nodes is microseconds in
   NumPy and needs no accelerator, no CUDA runtime and no framework
   version pinning.  Reporting an inference latency that depends on a
   GPU would not be an honest near-RT claim.
2. **Reproducibility.**  Every number here comes from a seeded
   ``np.random.Generator`` with no library-level nondeterminism, so a run
   is bit-reproducible on any machine.  That matters when the paper's
   central claim is a comparison between two methods.
3. **Dependencies.**  The project's ``requirements.txt`` is numpy, pandas,
   scipy, matplotlib, PyYAML and scikit-learn.  Adding a deep-learning
   framework for two small networks would be a heavier ask of anyone
   reproducing the work than writing the twelve lines of backpropagation
   below.

The gradients are hand-derived and checked numerically by
``scripts/selftest.py`` (finite differences, relative error < 1e-5), so
"hand-written" does not mean "unverified".

DESIGN
------
Every layer implements ``forward(x)`` (caching what backward needs) and
``backward(grad)`` (returning the gradient with respect to its input and
accumulating parameter gradients).  ``Module.parameters()`` returns a flat
list of ``Param`` objects that an optimiser can walk.  There is no graph,
no tape and no lazy evaluation: the call order IS the graph.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class Param:
    name: str
    value: np.ndarray
    grad: np.ndarray = field(init=False)

    def __post_init__(self):
        self.grad = np.zeros_like(self.value)

    def zero_grad(self) -> None:
        self.grad[...] = 0.0


class Module:
    def parameters(self) -> List[Param]:
        out: List[Param] = []
        for v in vars(self).values():
            if isinstance(v, Param):
                out.append(v)
            elif isinstance(v, Module):
                out.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, Module):
                        out.extend(x.parameters())
                    elif isinstance(x, Param):
                        out.append(x)
        return out

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def forward(self, x):
        raise NotImplementedError

    def backward(self, g):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    # ---- persistence --------------------------------------------------
    def state_dict(self) -> Dict[str, List]:
        return {p.name: p.value.tolist() for p in self.parameters()}

    def load_state_dict(self, blob: Dict[str, List]) -> None:
        byname = {p.name: p for p in self.parameters()}
        for k, v in blob.items():
            if k in byname:
                arr = np.asarray(v, dtype=float)
                if arr.shape == byname[k].value.shape:
                    byname[k].value[...] = arr


# ---------------------------------------------------------------------------
class Linear(Module):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator,
                 name: str = "fc", gain: float = 1.0, bias: bool = True):
        limit = gain * math.sqrt(2.0 / max(n_in, 1))
        self.W = Param(f"{name}.W", rng.normal(0.0, limit, (n_in, n_out)))
        self.b = Param(f"{name}.b", np.zeros(n_out)) if bias else None
        self._x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        y = x @ self.W.value
        if self.b is not None:
            y = y + self.b.value
        return y

    def backward(self, g: np.ndarray) -> np.ndarray:
        x = self._x
        self.W.grad += x.T @ g
        if self.b is not None:
            self.b.grad += g.sum(axis=0)
        return g @ self.W.value.T


class ReLU(Module):
    def forward(self, x):
        self._m = x > 0
        return x * self._m

    def backward(self, g):
        return g * self._m


class Tanh(Module):
    def forward(self, x):
        self._y = np.tanh(x)
        return self._y

    def backward(self, g):
        return g * (1.0 - self._y ** 2)


class Sigmoid(Module):
    def forward(self, x):
        self._y = 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
        return self._y

    def backward(self, g):
        return g * self._y * (1.0 - self._y)


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, g):
        for l in reversed(self.layers):
            g = l.backward(g)
        return g


def mlp(sizes: Sequence[int], rng: np.random.Generator, name: str,
        act=ReLU, out_act: Optional[Module] = None) -> Sequential:
    layers: List[Module] = []
    for k in range(len(sizes) - 1):
        layers.append(Linear(sizes[k], sizes[k + 1], rng, f"{name}.{k}"))
        if k < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act)
    return Sequential(*layers)


# ---------------------------------------------------------------------------
class Adam:
    """Adam with decoupled gradient clipping, so one odd epoch cannot
    blow the policy up in an online setting."""

    def __init__(self, params: Iterable[Param], lr: float = 3e-4,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, clip: float = 5.0):
        self.params = list(params)
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.clip = float(clip)
        self.t = 0
        self.m = [np.zeros_like(p.value) for p in self.params]
        self.v = [np.zeros_like(p.value) for p in self.params]

    def step(self) -> float:
        self.t += 1
        total = math.sqrt(sum(float((p.grad ** 2).sum())
                              for p in self.params))
        scale = 1.0
        if self.clip > 0 and total > self.clip:
            scale = self.clip / (total + 1e-12)
        for k, p in enumerate(self.params):
            g = p.grad * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p.value -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return total

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


# ---------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -60, 60))
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-30)


def save_state(path, blobs: Dict[str, Dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(blobs), encoding="utf-8")


def load_state(path) -> Dict[str, Dict]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
class RunningNorm:
    """Online feature standardisation (Welford), frozen at evaluation.

    Without it an online policy sees inputs whose scale changes as the
    plant drifts, which is indistinguishable from the policy's own
    learning and makes an ablation uninterpretable.
    """

    def __init__(self, n: int, clip: float = 5.0):
        self.n = 0
        self.mean = np.zeros(n)
        self.m2 = np.ones(n)
        self.clip = clip
        self.frozen = False

    def update(self, x: np.ndarray) -> None:
        if self.frozen:
            return
        x = np.atleast_2d(x)
        for row in x:
            self.n += 1
            d = row - self.mean
            self.mean += d / self.n
            self.m2 += d * (row - self.mean)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sd = np.sqrt(np.maximum(self.m2 / max(self.n - 1, 1), 1e-8))
        return np.clip((x - self.mean) / sd, -self.clip, self.clip)

    def state_dict(self) -> Dict:
        return {"n": self.n, "mean": self.mean.tolist(),
                "m2": self.m2.tolist()}

    def load_state_dict(self, d: Dict) -> None:
        if not d:
            return
        self.n = int(d.get("n", 0))
        self.mean = np.asarray(d.get("mean", self.mean), dtype=float)
        self.m2 = np.asarray(d.get("m2", self.m2), dtype=float)
