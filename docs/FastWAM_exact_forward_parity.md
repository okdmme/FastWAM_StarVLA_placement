# FastWAM exact forward parity

This procedure validates the StarVLA FastWAM port against the official
FastWAM implementation at the level needed to make a bitwise claim.  It is
not a LIBERO closed-loop evaluation.

## What is compared

The default `core` run deliberately bypasses image and text encoding.  It
persists one fixed CPU-generated input bundle and gives its exact bytes to both
models:

- first-frame VAE latent;
- text context and context mask;
- normalized proprio state;
- initial action noise, scheduler timesteps, and scheduler deltas.

It also audits every `mot` and `proprio_encoder` checkpoint tensor after each
load, including shape, dtype, value equality, and parameter coverage.  The
job succeeds only when all audit reports and every traced intermediate tensor
are `torch.equal`.

This separation matters: the official FastWAM encoder uses DiffSynth while
the StarVLA adapter uses Diffusers.  Comparing raw image outputs first would
mix an encoder discrepancy with an implementation discrepancy in the FastWAM
core.

## Required layout

The StarVLA checkout contains the port and the checkpoint.  The official
FastWAM checkout must be the pinned official revision `45d8e1458921d83f8ad6cf9ce993d371208dabd0`.

```text
work/
├── starVLA/
│   ├── checkpoints/fastwam_release/libero_uncond_2cam224.pt
│   └── .venv-py311/
├── FastWAM/
└── fastwam_official_assets/
    └── DiffSynth-Studio/Wan-Series-Converted-Safetensors/
        └── Wan2.2_VAE.safetensors
```

`/home/anpan/WM/FastWAM` is already at that official revision.  The current
`/home/anpan/WM/checkpoints` directory is empty; the actual local checkpoint
is `/home/anpan/WM/starVLA/checkpoints/fastwam_release/libero_uncond_2cam224.pt`.

The official converted VAE is not currently present in the workspace.  Fetch
it once on ABCI (or copy a pre-fetched directory there).  For the raw-encoder
diagnostic below, download the text encoder as well.

```bash
cd ~/work
source starVLA/.venv-py311/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="SereneC/wan-series-checkpoint",
    revision="fec1e03",
    local_dir="fastwam_official_assets",
    allow_patterns=[
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
    ],
)
PY
```

Use one Python environment for both implementations.  Do not create one
environment per repository or let the two projects install different PyTorch
builds: the runner imports both code paths in the same Python process so each
uses the same CUDA, cuDNN, and PyTorch kernels.  Installing the official source
without its dependency resolver changing the StarVLA environment is sufficient.

```bash
cd ~/work/starVLA
source .venv-py311/bin/activate
python -m pip install -e ../FastWAM --no-deps
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

If that environment cannot import either repository, create a clean Python
3.11 environment with the ABCI CUDA-compatible PyTorch build, install
StarVLA's requirements, then install the official source with `--no-deps`.
Record the resulting package versions with the parity evidence; the runner
already records Python, PyTorch, CUDA, and cuDNN versions in both manifests.

## ABCI command

Edit `<ABCI_GROUP>` in `docs/abci_fastwam_exact_core_parity.pbs`, then submit
from the StarVLA checkout.

First, run the preflight on the login node.  It checks the required Python
imports, the pinned FastWAM revision, checkpoint/stat files, the converted
official VAE, and imports the StarVLA framework registry without allocating a
model.  It exits non-zero and lists every detected issue, so do not submit a
GPU job until its JSON output says `"ok": true`.

```bash
source .venv-py311/bin/activate
python docs/fastwam_parity_preflight.py
```

The supplied PBS loads ABCI's `python/3.12/3.12.9` module before activating
the venv.  Keep that line in place: a venv created from a module-provided
Python needs the module's `libpython` on the compute node as well.

```bash
cd ~/work/starVLA
qsub docs/abci_fastwam_exact_core_parity.pbs
qstat
tail -f abci_fastwam_exact_core_parity.log
```

If the sibling paths differ, pass them explicitly at submission time.

```bash
qsub -v FASTWAM_OFFICIAL_DIR=/path/to/FastWAM,FASTWAM_OFFICIAL_ASSETS_DIR=/path/to/fastwam_official_assets \
  docs/abci_fastwam_exact_core_parity.pbs
```

Success means the log ends with a JSON object whose `exact` field is `true`.
The evidence bundle is written to
`playground/fastwam_parity/exact_core_seed7/` and includes:

- `official_checkpoint_audit.json` and `starvla_checkpoint_audit.json`;
- `comparison.json`, per-stage bitwise comparison results;
- `anchor/fixed_core_inputs.pt`, the immutable common input;
- `official_manifest.json` and `starvla_manifest.json`, with raw-byte SHA-256
  digests; and
- `fastwam_parity_results.zip`.

## Raw encoder diagnostic

Run this only after core parity passes.  It diagnoses VAE/tokenizer/text
encoder/preprocessing differences; it should not be used as evidence of core
parity because the two public implementations use different encoder stacks.

```bash
cd ~/work/starVLA
source .venv-py311/bin/activate
export PYTHONHASHSEED=7 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=0 TOKENIZERS_PARALLELISM=false FLASH_ATTENTION_FORCE_DISABLE=1
python -m examples.simBenchmarks.LIBERO.eval_files.fastwam_parity.colab_orchestrator \
  --runtime-dir playground/fastwam_parity/raw_seed7 \
  --starvla-dir "$PWD" \
  --official-dir ../FastWAM \
  --official-assets-dir ../fastwam_official_assets \
  --input-mode raw \
  --seed 7 \
  --num-inference-steps 10 \
  --dtype bfloat16
```

For raw mode, the official text asset listed above and StarVLA's local Wan2.2
Diffusers `tokenizer/`, `text_encoder/`, and `vae/` directories are required.
Do not add `--require-exact` to this diagnostic unless the two encoder weights
and implementations have first been proven bitwise identical.
