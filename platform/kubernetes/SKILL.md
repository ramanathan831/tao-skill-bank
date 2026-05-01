---
name: kubernetes
description: "Kubernetes GPU execution platform for TAO SDK jobs. Use when the user wants to run TAO workloads on Kubernetes or k8s."
---

# Kubernetes Platform

Kubernetes runs TAO jobs as Kubernetes Jobs on a reachable GPU cluster. The SDK
authenticates through kubeconfig or an in-cluster service account.

## Credentials

Do not ask for Lepton, Brev, or SLURM credentials for Kubernetes runs.

Required platform credentials:

- None, if the current environment already has a usable kubeconfig or
  in-cluster service account.

Optional platform settings:

- `KUBECONFIG`: non-default kubeconfig path.
- `TAO_K8S_NAMESPACE`: namespace, default `default`.
- `TAO_K8S_CONTEXT`: kubeconfig context name.
- `NGC_KEY`: image pull access for private `nvcr.io` images.
- `ACCESS_KEY`, `SECRET_KEY`, `S3_ENDPOINT_URL`, `S3_BUCKET_NAME`,
  `CLOUD_REGION`: ask only when the selected workflow uses `s3://` inputs or
  outputs.
- `HF_TOKEN`: ask only when the selected model requires HuggingFace access.

## Launch Preflight

Before generating specs or creating Jobs:

1. Verify `kubectl` can reach the selected context and namespace.
2. Verify the account can create Jobs in `TAO_K8S_NAMESPACE`.
3. Verify the cluster has allocatable `nvidia.com/gpu` capacity, or get explicit
   proof from the cluster admin when node-list RBAC is restricted.
4. For `s3://` datasets/results, verify `ACCESS_KEY` and `SECRET_KEY` are set
   and the exact paths are readable with `aws s3 ls`.
5. For PVC or mounted filesystem paths, require proof that the path is mounted
   into the job container. Do not accept an agent-host local path as proof.
6. Verify model-specific credentials such as `HF_TOKEN` before launch.

## Notes

- The cluster must have NVIDIA GPU Operator or equivalent device plugin support
  for `nvidia.com/gpu`.
- Use `kubernetes` as the canonical backend type. Accept `k8s` and `local-k8s`
  as aliases.
- Prefer `scripts/list_tao_platforms.py --platform kubernetes` when deciding
  what credentials to ask for.
