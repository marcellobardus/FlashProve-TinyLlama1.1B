# Reproduce

## RunPod pod

Heavy work (download, conversion, calibration, model-dependent tests, quality evaluation) runs on a RunPod CPU
pod, not on the shared box (work/DECISIONS.md D7).

- Pod roensk5w2gn6am (created 2026-11-04 22:08:55 Almaty, data center EUR-IS-1; it replaced l925odnw8dlhn1, which could not start for lack of host capacity and was terminated), flavor cpu3g-8-32 (8 vCPU, 32 GB RAM), secure cloud, 0.32 USD/hr as reported by the API.
- Image runpod/pytorch:1.5.1-cu1281-torch280-ubuntu2204.
- 50 GB container disk, no volume, port 22/tcp exposed.
- Access by SSH with a dedicated key. No secrets, host, port or API key are kept in this repository.
- The pod is stopped when idle.

Unverified caveats: stopping the pod may wipe the container disk, since there is no volume, so anything on it
must be treated as lost after a stop. The public SSH port may change after a restart, so read the current port
from the pod status before connecting.

Steps: start the pod through the RunPod console or API with your own key held in an environment variable. Read
the pod status to get the current public SSH address and port. Connect with the dedicated key. Clone the
repository, check out feature/tinyllama-integer-audit and run the documented commands of work/TASKS.md for the
task at hand. Copy any results you need off the pod before stopping it. Stop the pod when it is idle.

## Environment used for the first real run (2026-11-04)

Python 3.13.18 in a uv venv (check_inputs.py needs 3.13), torch 2.9.1+cpu (installed from the PyTorch CPU wheel index), transformers 5.22.1, tokenizers 0.23.3, numpy 2.5.4, pyarrow 26.0.1, huggingface_hub 2.4.0, safetensors 0.8.0. The checkpoint is fetched at the pinned revision with the Hugging Face CLI. With transformers 5.22.1, P5 failed (work/DECISIONS.md D29); the next run uses a transformers 4.x release and records the version. `time` was installed with apt to measure peak memory.

Observed on the pod: the container disk is wiped when the pod is stopped (no volume), the public SSH port changes, boot to SSH took up to about 13 minutes, a stopped pod may fail to start when its host has no free capacity (the first pod never started again), and creating a new pod may work on the first request in the same data center. Copy results off the pod before stopping it.
