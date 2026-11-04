# Reproduce

## RunPod pod

Heavy work (download, conversion, calibration, model-dependent tests, quality evaluation) runs on a RunPod CPU
pod, not on the shared box (work/DECISIONS.md D7).

- Pod id l925odnw8dlhn1, flavor cpu3g-8-32 (8 vCPU, 32 GB RAM), secure cloud, 0.32 USD/hr as reported by the API.
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
