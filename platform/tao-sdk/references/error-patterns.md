# TAO SDK Error Patterns

Common SDK failure modes and direct remediation.

## Contents

- Credential errors.
- Missing output/input wrapping.
- Pending jobs and image pulls.
- Dataset URI formatting.
- Brev capacity issues.

## Patterns

**`CredentialError: Missing LEPTON_WORKSPACE_ID`**: env var not loaded. Run `source ~/.config/tao/.env` or check the SessionStart hook fired.

**`CredentialError: S3_BUCKET_NAME env var required`**: any `inputs` or `outputs` argument needs S3 credentials. Set `S3_BUCKET_NAME`, `ACCESS_KEY`, `SECRET_KEY` and `S3_ENDPOINT_URL` for non-AWS.

**TAO crash: `You need to set ... results_dir`**: `build_entrypoint` was called without `outputs=action_cfg["outputs"]`. Same root cause if S3 input URIs are not downloaded: `inputs=action_cfg["inputs"]` was omitted. Mirror both from `skill_info.yaml`.

**Job stuck in `Pending` (Lepton)**: call `get_job_replicas(job_id)` and inspect `readiness_issue`. Most common causes are image pull delay or `ConfigError` on a bad node; cancel and resubmit when needed.

**`Image pull failed`**: `NGC_KEY` is invalid or expired. Refresh the key and resubmit.

**Double slash in S3 URI**: use `dataset_uri.rstrip("/")` before concatenating, or use `os.path.join`.

**Brev instance will not start**: GPU type unavailable in the user's region. Try a different `gpu_type` or wait.
