# TAO SDK Platform Notes

Read this when selecting platform-specific SDK kwargs or credential prompts.

## Contents

- Lepton notes.
- Brev notes.
- SLURM notes.
- Kubernetes notes.
- Local Docker notes.

## Lepton

- Jobs run as containers on DGX Cloud.
- NFS/Lustre mounts auto-detected from the node group; the SDK builds the appropriate `Mount` objects.
- `gpu_count` resolves to a Lepton resource shape; or pass `dedicated_node_group="<name>"` for guaranteed allocation.
- `num_nodes=N` (N>1) enables distributed training.

## Brev

- Jobs run on GPU instances via `brev exec`.
- No shared storage; use S3.
- Pass `instance_id="<id>"` in kwargs to reuse an existing instance and skip boot time.
- Pass `gpu_type="L40S"` to control instance class for ephemeral instances.
- Use `sdk.delete_instance(instance_id)` when done with an ephemeral instance.

## SLURM

- Jobs submit over SSH to a login node with `sbatch` and run containers through Pyxis/Enroot `srun --container-image`.
- Use the platform helper output to ask only for SLURM credentials and storage settings. Do not ask for Lepton, Brev, or Kubernetes credentials.
- Dataset paths must be visible from the cluster job, usually absolute Lustre or shared filesystem paths; do not pass agent-host local paths to SLURM jobs.
- Use the packaged SLURM runtime defaults unless the user gives a validated override. For the common `polar,polar3,polar4,grizzly` queues, prefer the four-hour default rather than generating 12-hour wrappers.

## Kubernetes

- Jobs run as Kubernetes Jobs on a configured GPU cluster.
- Auth uses kubeconfig (`KUBECONFIG` or `~/.kube/config`) or an in-cluster service account.
- Requires NVIDIA GPU Operator or equivalent `nvidia.com/gpu` device plugin.
- Do not ask for Lepton, Brev, or SLURM credentials for Kubernetes runs.
- A local path on the agent host is not proof that the path is mounted inside the job pod.

## Local Docker

- Jobs run on the local Docker daemon host.
- Multi-node is not supported; multi-GPU on the local host is supported.
- Verify local dataset paths, Docker daemon access, and NVIDIA runtime before generating or launching runner artifacts.
