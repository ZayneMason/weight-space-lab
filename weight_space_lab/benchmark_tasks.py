"""Original synthetic editing tasks; operator-only generators and feasible controls."""
import numpy as np

FAMILIES = ("threshold", "slope", "routing", "hidden")


def _predict(arrays, architecture, inputs):
    x = inputs.astype(np.float64)
    for layer in architecture["layers"]:
        x = x @ arrays[layer["weight"]].astype(np.float64).T + arrays[layer["bias"]]
        if layer["activation"] == "relu":
            x = np.maximum(x, 0)
    return x.argmax(axis=1)


def _dataset(rng, source, teacher, architecture, per_group, prefix, dimensions):
    groups = {"repair": [], "protected": []}
    for _ in range(100):
        inputs = rng.uniform(-2, 2, size=(2048, dimensions))
        expected = _predict(teacher, architecture, inputs)
        original = _predict(source, architecture, inputs)
        for x, y, old in zip(inputs, expected, original):
            group = "protected" if y == old else "repair"
            if len(groups[group]) < per_group:
                groups[group].append({"id": f"{prefix}-{group}-{len(groups[group])}", "x": x.tolist(), "y": int(y)})
        if all(len(rows) == per_group for rows in groups.values()):
            return {"split": "practice" if prefix == "practice" else "heldout",
                    "examples": [row for rows in groups.values() for row in rows]}
    raise ValueError("Task could not supply the predeclared repair/protected strata")


def generate(family, seed, heldout_seed):
    if family not in FAMILIES or type(seed) is not int or type(heldout_seed) is not int or min(seed, heldout_seed) < 0:
        raise ValueError("Use a supported task family and nonnegative integer seeds")
    rng = np.random.default_rng(seed)
    a, b, threshold = rng.uniform(.7, 1.5), rng.uniform(-.6, .6), rng.uniform(.2, .4)
    dimensions = 1 if family in ("threshold", "slope") else 2
    architecture = {"layers": [{"weight": "weight", "bias": "bias", "activation": "linear"}]}
    teacher = {"weight": np.array([[-a], [a]], dtype=np.float32),
               "bias": np.array([0, -2*a*threshold], dtype=np.float32)}
    tensor, index = ("weight", 1) if family == "slope" else ("bias", 1)
    if family == "routing":
        teacher["weight"] = np.array([[-a, -b], [a, b]], dtype=np.float32)
        tensor, index = "weight", 3
    elif family == "hidden":
        architecture = {"layers": [
            {"weight": "hidden_weight", "bias": "hidden_bias", "activation": "relu"},
            {"weight": "weight", "bias": "bias", "activation": "linear"}]}
        teacher = {"hidden_weight": np.array([[a, b], [-a, -b]], dtype=np.float32),
                   "hidden_bias": np.zeros(2, dtype=np.float32),
                   "weight": np.array([[-1, 1], [1, -1]], dtype=np.float32),
                   "bias": np.array([0, -4*a*threshold], dtype=np.float32)}
        tensor, index = "hidden_weight", 1
    source = {name: values.copy() for name, values in teacher.items()}
    source[tensor].flat[index] = -.5*a if family == "slope" else teacher[tensor].flat[index] + rng.choice([-1, 1]) * (1.6 if dimensions == 2 else .8)
    # Sequential independent generators; paired models receive these same retained draws.
    practice = _dataset(rng, source, teacher, architecture, 16, "practice", dimensions)
    heldout = _dataset(np.random.default_rng(heldout_seed), source, teacher, architecture, 128, "final", dimensions)
    return {"family": family, "seed": seed, "heldout_seed": heldout_seed,
            "arrays": source, "architecture": architecture, "practice": practice, "heldout": heldout,
            "editable": {"tensor": tensor, "flat_indices": [index], "max_changed_entries": 1},
            "control_value": float(teacher[tensor].flat[index])}
