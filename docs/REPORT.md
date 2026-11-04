# Weight Space Lab: Implementation and Results

**Date:** November 4, 2026

**Status:** Results snapshot; full MATH-500 comparison pending

**Project:** Weight Space Lab

## Executive summary

Weight Space Lab provides a working research prototype for inspecting, modifying, and evaluating neural-network weights. The project now combines a reusable checkpoint toolkit, a previously completed small-network demonstration, and a functioning GPU evaluation route for a language model.

A verified result is an exact reconstruction of the published Qwen weight edit: **130,188 BF16 entries changed across two tensors, with both expected edited-tensor hashes matching and all 318 other exported tensors bitwise unchanged.** A paired language-model pilot is complete; the full evaluation has been launched.

The pilot produced **one correct answer out of eight samples for the original model and six out of eight for the edited model**. This is encouraging preliminary evidence from two questions. It is not a completed MATH-500 benchmark result.

The framework is at the research-prototype stage. Its editing and measurement tools work; the controlled loop for discovering fresh language-model edits remains to be built.

## Purpose and scope

Weight space programming uses deliberate changes to a model's numerical parameters to alter its behavior. In the intended workflow, an agent inspects a separate model, forms a hypothesis, writes a candidate patch, measures its behavior on practice tasks, and freezes a candidate before final evaluation. Codex supplies the reasoning and orchestration; the local checkpoint is the object being edited.

This report summarizes the checkpoint toolkit, recorded Breakout experiments, and the Qwen reconstruction and pilot. Breakout results are documented in the project results record; Qwen verification artifacts are included under `evidence/qwen/`.

## Verified accomplishments

### 1. Established reusable checkpoint-editing tools

The existing toolkit can inspect tensor names, shapes, types and statistics; read bounded tensor slices; and apply indexed set, add, scale, or direction-removal operations. It writes a separate candidate and records the source hash, patch rationale, exact operations, and actual changed-entry count.

The original checkpoint remains available for rollback. Validation checks include source hashes, edit budgets, finite values, index bounds, supported operations, and preservation of unedited values. The toolkit also includes a dense-network evaluator and paired comparison reports listing fixed and regressed examples.

The generic editor currently supports individual NPZ or SafeTensors files with floating-point edits in F16/F32/F64. The new Qwen route handles its published BF16 edit separately; broad BF16, quantized, and arbitrary multi-shard editing support is not yet unified into the generic editor.

### 2. Preserved a plugin foundation for Codex and DeepSeek Harness

The plugin contains shared skill resources, a Codex compatibility manifest, and a DSH npm bundle. Both Node tests passed, checking DSH skill registration/disposal and common resource resolution across the two hosts.

Isolated-profile DSH validation is documented in the project history. Active-profile installation and graphical plugin discovery have not been verified. The Qwen extension currently runs alongside the plugin through command-line tools.

### 3. Built and validated a usable GPU environment

The Qwen evaluation environment was validated on an NVIDIA RTX 3090 with 24 GB VRAM under Ubuntu 24.04 in WSL.

The working stack includes PyTorch 2.14.0+cu130, vLLM 0.33.0, Transformers 5.20.2, and the pinned mathematical grader. Build dependencies, matching CUDA compiler/toolkit versions, and linker paths were configured for GPU inference. The complete Python package set is recorded in a lockfile.

After enabling decode CUDA graphs and the working sampling configuration, the short baseline check generated approximately **950 output tokens per second**, compared with approximately 100 in the earlier eager/native-sampler check. These are short-check measurements, not sustained full-benchmark performance estimates.

### 4. Reconstructed the published Qwen edit exactly

The reconstruction uses the pinned Qwen3.5-0.8B release and the published p50c edit. It preserves the released text-only configuration and chat template.

| Exported tensor | BF16 entries changed | Verification |
|---|---:|---|
| Layer 15 MLP down-projection | 65,260 | Base and edited tensor hashes matched |
| Layer 19 MLP down-projection | 64,928 | Base and edited tensor hashes matched |
| **Total** | **130,188** | **Expected count matched** |

An independent audit compared both complete exported checkpoints. Their inventories contain 320 tensors, exactly two have the expected changes, and the other 318 are bitwise identical. This establishes that the evaluated candidate contains the intended published edit without additional exported-tensor changes.

### 5. Implemented a reproducible language-model evaluation route

The new runner prepares frozen checkpoints and a pinned MATH-500 dataset, renders thinking-enabled prompts, generates four samples per problem, and grades the final boxed answer with math-verify 0.9.0.

For every completed problem it retains the prompt, prompt token IDs, all four generated texts, output token IDs and counts, seeds, finish reasons, truncation flags, extracted answers, and grades. Job metadata binds results to the input hashes, source version, installed packages, and engine settings. Atomic writes preserve completed problems for resumption, and incompatible jobs are rejected instead of mixed together.

The comparison reports average accuracy across all samples and token-limit frequency. Its paired bootstrap keeps the four samples from each problem together as one uncertainty unit.

### 6. Passed implementation checks

| Test group | Result |
|---|---|
| Existing Python toolkit and Breakout tests | 21 passed; 1 reference-asset integration test skipped |
| New Qwen tests | 8 passed |
| Node plugin tests | 2 passed |

The Qwen tests cover exact BF16 reconstruction; rejection of incorrect hashes, counts, and indices; symbolic grading; agreement with the released answer extractor; and paired-comparison validation. The skipped existing test requires Breakout reference assets not bundled in the repository. The complete exported-checkpoint audit passed separately.

## Experimental results

### Breakout

The project results record documents the following MinAtar Breakout measurements for a 20,486-parameter network:

| Network | Mean score | Standard error | Episodes |
|---|---:|---:|---:|
| Published research reference | 517.411 | 7.417 | 1,000 |
| Released weights evaluated locally | 502.880 | 7.747 | 1,000 |
| Fresh agent-written network evaluated locally | 511.530 | 5.547 | 1,000 |

The recorded fresh attempt used one tool-free Astra response, with the final network frozen before evaluation. It demonstrates successful direct weight writing for this small task. The fresh network's 8.65-point advantage over the local replay has a paired-difference standard error of 9.00, so it does not establish an improvement over the released network.

**Evidence boundary:** these measurements are documented in `RESULTS.md`; raw Breakout transcripts and episode logs are not included in this repository. A fresh conversation cannot rule out prior exposure in the editor model's training data.

### Qwen pilot at the published token limit

The paired pilot evaluated the first two MATH-500 problems with four samples per problem per checkpoint. Both checkpoints used thinking mode, a 32,768-new-token ceiling, temperature 0.6, top-p 0.95, top-k 20, and presence penalty 1.5.

| Measure | Original Qwen | Published p50c edit |
|---|---:|---:|
| Problems | 2 | 2 |
| Generated samples | 8 | 8 |
| Correct samples | 1/8 (12.5%) | 6/8 (75.0%) |
| Samples reaching the token ceiling | 8/8 (100%) | 1/8 (12.5%) |
| Total generated tokens | 262,144 | 134,717 |

The pilot successfully exercised reconstruction, inference, evidence capture, grading, and paired comparison at the published generation limit. On these two questions, the edited checkpoint produced more correct answers and fewer capped responses. The sample is too small to support a general accuracy claim or a reliable estimate of benchmark improvement.

The pilot evaluates the published p50c edit. Fresh Qwen edit discovery is not implemented.

## Framework maturity

| Component | Current maturity |
|---|---|
| Checkpoint inspection and bounded editing | Working, with documented format limits |
| Candidate preservation and provenance | Working |
| Small-network direct weight writing | Demonstrated in the recorded Breakout experiment |
| Published Qwen BF16 reconstruction | Verified exactly |
| Language-model inference and evaluation | Working; pilot completed |
| Complete paired MATH-500 measurement | In progress at the report snapshot |
| Fresh language-model edit discovery | Not implemented |
| Unified interface across architectures | Not implemented |

The project has established useful, tested building blocks for a weight space programming framework. Its current scope is a research toolkit with task-specific adapters. A general autonomous framework will require a controlled editor, practice-only feedback, candidate-selection rules, and a common interface connecting these pieces.

## Evidence and reproducibility

The research source is pinned to commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`; the Qwen model revision is `2fc06364715b967f1860aea9cf38778875588b17`; and the MATH-500 dataset revision is `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.

The local evaluator groups four problems per call and assigns each problem a seed equal to its dataset ordinal. The released evaluator generates a whole shard with seed zero. Local hardware and inference-package versions are recorded separately. These differences make the experiment a documented statistical reproduction attempt rather than an identical replay of the original random trajectories.

Repository evidence:

- [Earlier Breakout results](../RESULTS.md)
- [Qwen reconstruction provenance and hashes](../evidence/qwen/reconstruction.json)
- [Independent exported-checkpoint audit](../evidence/qwen/export-audit.json)
- [Completed Qwen pilot comparison](../evidence/qwen/pilot-comparison.json)
- [Implementation and run instructions](../qwen/README.md)

The reconstruction record retains source URLs, asset hashes, checkpoint hashes, and edited-tensor verification. Checkpoints, benchmark questions, generated answers, and machine-specific runtime state are excluded from the public repository.

Research context: [published Breakout experiment](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch) and [published Qwen experiment](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math).

## Remaining work

The full paired MATH-500 comparison was in progress at the report snapshot. It requires 500 problems, four samples per problem, and two frozen checkpoints: 4,000 generations in total. Final accuracy and truncation results are not yet available.

The next milestones are to finish and analyze that comparison, build a CPU editing sandbox and a metered practice-only forward interface, and test a fresh agent-discovered Qwen edit under an explicit budget. Candidate selection must remain separate from final evaluation. The checkpoint toolkit, model adapters, and experiment controller can then be consolidated under one reusable interface.

**Established achievement:** the project can preserve, reconstruct, execute, and evaluate deliberate neural-weight changes with recorded provenance. It has a documented small-network demonstration and a directly verified language-model pilot, providing a concrete foundation for the broader framework.
