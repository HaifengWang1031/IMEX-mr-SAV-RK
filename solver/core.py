"""Small contracts shared by schemes, adaptive algorithms and the driver."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from types import MappingProxyType
import uuid
import numpy as np


@dataclass(frozen=True)
class State:
    fields: dict
    aux: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.fields:
            raise ValueError("At least one named physical field is required")
        fields, auxiliary = {}, {}
        for name, value in self.fields.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Field names must be nonempty strings")
            array = np.array(value, copy=True)
            if array.dtype.kind not in "fci":
                raise ValueError("Physical fields must be numeric arrays")
            array.setflags(write=False)
            fields[name] = array
        for name, value in self.aux.items():
            if not isinstance(name, str) or not name or not np.isscalar(value) or not np.isrealobj(value):
                raise ValueError("Auxiliary variables must be named real scalars")
            auxiliary[name] = float(value)
        object.__setattr__(self, "fields", MappingProxyType(fields))
        object.__setattr__(self, "aux", MappingProxyType(auxiliary))

    def finite(self):
        return all(np.isfinite(a).all() for a in self.fields.values()) and all(np.isfinite(v) for v in self.aux.values())

    def same_layout(self, other):
        return (self.fields.keys() == other.fields.keys() and self.aux.keys() == other.aux.keys()
                and all(a.shape == other.fields[k].shape for k, a in self.fields.items()))


@dataclass(frozen=True)
class History:
    """Only accepted states; chronological order, with the original accepted step sizes."""
    states: tuple
    times: tuple
    steps: tuple = ()

    @property
    def state(self):
        return self.states[-1]

    @property
    def time(self):
        return self.times[-1]

    def commit(self, state, time, dt, size):
        return History((*self.states, state)[-size:], (*self.times, time)[-size:],
                       (*self.steps, dt)[-(size-1):] if size > 1 else ())


@dataclass(frozen=True)
class Trial:
    state: State
    embedded_fields: dict | None = None


@dataclass(frozen=True)
class Assessment:
    trial: Trial
    accept: bool
    next_dt: float
    metrics: dict = field(default_factory=dict)


class Scheme:
    """New schemes implement step(model, accepted_history, dt), without mutating history.

    Mutable caches may hold coefficients, never an uncommitted numerical history.
    Multistep schemes also provide startup(model, history, dt).
    """
    name = "unnamed"
    order = None
    history_size = 1
    variable_step = False
    embedded = False
    auxiliary_defaults = {}

    def initialize(self, state):
        if state.aux.keys() - self.auxiliary_defaults.keys():
            raise ValueError(f"Unexpected auxiliary variables for {self.name}")
        return State(state.fields, self.auxiliary_defaults | dict(state.aux))

    def step(self, model, history, dt, *, estimate=False):
        raise NotImplementedError

    def startup(self, model, history, dt):
        raise NotImplementedError(f"{self.name} needs a startup scheme")


class AdaptiveAlgorithm:
    """A complete algorithm owns its trial construction, metrics and controller history."""
    def validate(self, scheme, state):
        pass

    def reset(self):
        pass

    def attempt(self, model, scheme, history, dt):
        raise NotImplementedError

    def on_accept(self, assessment, forced):
        pass

    def on_reject(self, assessment):
        pass


@dataclass
class Result:
    times: np.ndarray
    steps: np.ndarray
    auxiliary: dict
    diagnostics: dict
    requested_times: np.ndarray
    snapshot_times: np.ndarray
    snapshot_indices: np.ndarray
    snapshot_fields: dict
    final_state: State
    stats: dict
    metadata: dict = field(default_factory=dict)
    failed_state: State | None = None

    @property
    def actual_snapshot_times(self):
        """One value per request; NaN means an unresolved request in a failed run."""
        return np.array([self.snapshot_times[i] if i >= 0 else np.nan for i in self.snapshot_indices])

    def save(self, path, metadata=None):
        """NPZ, no pickle; preserve existing evidence and atomically publish new output."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {"schema": 1, "stats": self.stats, "metadata": self.metadata | (metadata or {}),
                "fields": list(self.final_state.fields), "auxiliary": list(self.auxiliary),
                "diagnostics": list(self.diagnostics), "final_aux": dict(self.final_state.aux),
                "failed_aux": None if self.failed_state is None else dict(self.failed_state.aux)}
        arrays = {"metadata": np.array(json.dumps(info)), "times": self.times, "steps": self.steps,
                  "requested_times": self.requested_times, "snapshot_times": self.snapshot_times,
                  "snapshot_indices": self.snapshot_indices}
        for index, name in enumerate(info["fields"]):
            arrays[f"snapshots_{index}"] = self.snapshot_fields[name]
            arrays[f"final_{index}"] = self.final_state.fields[name]
            if self.failed_state is not None:
                arrays[f"failed_{index}"] = self.failed_state.fields[name]
        for category in ("auxiliary", "diagnostics"):
            for index, name in enumerate(info[category]):
                arrays[f"{category}_{index}"] = getattr(self, category)[name]
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            # link() fails if the destination exists, including concurrent writers.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as source:
            arrays = {k: source[k].copy() for k in source.files}
        info = json.loads(str(arrays.pop("metadata")))
        if info["schema"] != 1:
            raise ValueError("Unsupported result schema")
        fields = info["fields"]
        failed = None if info["failed_aux"] is None else State(
            {n: arrays[f"failed_{i}"] for i, n in enumerate(fields)}, info["failed_aux"])
        return cls(times=arrays["times"], steps=arrays["steps"],
                   auxiliary={n: arrays[f"auxiliary_{i}"] for i, n in enumerate(info["auxiliary"])},
                   diagnostics={n: arrays[f"diagnostics_{i}"] for i, n in enumerate(info["diagnostics"])},
                   requested_times=arrays["requested_times"], snapshot_times=arrays["snapshot_times"],
                   snapshot_indices=arrays["snapshot_indices"],
                   snapshot_fields={n: arrays[f"snapshots_{i}"] for i, n in enumerate(fields)},
                   final_state=State({n: arrays[f"final_{i}"] for i, n in enumerate(fields)}, info["final_aux"]),
                   stats=info["stats"], metadata=info["metadata"], failed_state=failed)


class IntegrationError(RuntimeError):
    def __init__(self, message, result):
        super().__init__(message)
        self.result = result
