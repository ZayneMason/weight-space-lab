# CLI reference

Run with `uv run --script /absolute/path/to/scripts/weight_lab.py COMMAND` or Python with NumPy 2.5.4 and SafeTensors 0.8.0. Commands emit JSON; errors go to stderr and return exit code 2. Output files/directories must not exist. Parent directories must already exist.

## Inspect and read

```sh
weight_lab.py inspect model.safetensors --max-mib 512
weight_lab.py tensor model.safetensors model.layers.15.mlp.down_proj.weight --start 0 --count 64
```

`tensor` reads C-order flat values, at most 4096 per call. `inspect` reports tensor names, shapes, dtypes, min/max/mean and zero fractions. Both load the checkpoint locally; their output is bounded but the file is not lazily streamed. The default byte budget is 512 MiB; memory use can be several times that. NPZ decompressed bytes are checked before loading. F16/F32/F64 and integer storage are accepted for inspection; edits support floats only. BF16, FP8, quantized formats and checkpoint directories are not supported. Work with an explicit compatible copy if needed; conversion itself can change behavior and needs a fresh baseline.

## Patch

```json
{
  "base_sha256": "COPY_THE_SHA256_FROM_INSPECT",
  "reason": "Hypothesis and expected behavioral change",
  "operations": [
    {"op": "scale", "tensor": "weight", "indices": [0, 1], "values": [-1, -1]}
  ]
}
```

```sh
weight_lab.py patch original.npz patch.json candidate --max-fraction 0.01
```

`set`, `add`, and `scale` use unique nonnegative C-order flat indices and one finite value per index. Operations apply in order. Unknown fields fail validation. The limit counts the number of stored values that differ from the original after all operations, divided by all elements in that checkpoint file (including unchanged tensors). It is not a magnitude bound or a whole-model fraction when the input is only one shard. No-op patches fail.

For a 2D `[output, input]` matrix, this operation subtracts the projection along a direction from selected columns:

```json
{"op":"remove_direction","tensor":"weight","columns":[0,2],"direction":[1,0,0],"strength":1}
```

The direction must match the output dimension. The tool normalizes it and computes `W[:, columns] -= strength * outer(d, d @ W[:, columns])`. Strength is in `[0,1]`. This mathematical operation does not establish that the direction corresponds to a behavior; that requires experimental evidence.

Outputs: `model.npz` or `model.safetensors`, `patch.json`, and `receipt.json`. SafeTensors metadata is retained. Shape, dtype and unedited tensor values are preserved. The source is never overwritten; choosing it again is rollback. No model config or tokenizer is copied, and no checkpoint is deployed.

## Dense-network evaluation

Architecture JSON:

```json
{"layers":[{"weight":"weight","bias":"bias","activation":"linear"}]}
```

Each layer uses `x @ weight.T + bias`; allowed activations are `linear`, `relu`, `tanh`. Weights are `[output,input]` and biases `[output]`. Last-layer outputs are >=2 class logits, classified by argmax. Computation uses float64, independent of tensor storage dtype; this is a reference evaluator, not bit-identical PyTorch inference.

Dataset JSON:

```json
{"split":"heldout","examples":[{"id":"negative-1","x":[-2.5],"y":0},{"id":"positive-1","x":[2.5],"y":1}]}
```

```sh
weight_lab.py evaluate-mlp original.npz architecture.json heldout.json --output baseline.json
weight_lab.py evaluate-mlp candidate/model.npz architecture.json heldout.json --output candidate-eval.json
weight_lab.py compare baseline.json candidate-eval.json
```

Comparison requires the same architecture bytes, dataset bytes, split, settings and example IDs/labels. It recomputes accuracy from predictions and lists fixed/regressed IDs. Dataset split labels are descriptive, not an access-control mechanism: the agent must keep held-out examples out of edit selection. Transformer evaluation requires an external model-specific harness; `compare` accepts only this toolkit's MLP report schema.

## Demo

```sh
weight_lab.py demo /absolute/path/to/new-demo-folder
```

Creates an inverted sign classifier, applies a prewritten two-weight sign flip, and scores six predefined examples. This verifies the toolkit end to end; it does not test an agent's ability to discover an edit or reproduce any research benchmark.
