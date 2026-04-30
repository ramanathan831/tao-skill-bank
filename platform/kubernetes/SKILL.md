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

## Notes

- The cluster must have NVIDIA GPU Operator or equivalent device plugin support
  for `nvidia.com/gpu`.
- Use `kubernetes` as the canonical backend type. Accept `k8s` and `local-k8s`
  as aliases.
- Prefer `scripts/list_tao_platforms.py --platform kubernetes` when deciding
  what credentials to ask for.
