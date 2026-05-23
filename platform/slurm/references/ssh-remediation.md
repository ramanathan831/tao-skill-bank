# SLURM SSH Remediation

Use this when the SLURM preflight cannot reach a login host with passwordless
SSH.

## Contents

- User prompt for `SSH_KEY_PATH`.
- Passwordless SSH setup commands.
- Default result path behavior.

## Prompt

```text
SLURM is blocked on passwordless SSH. Please provide:

SSH_KEY_PATH=/path/to/private_key

If you have not set up passwordless access yet:
1. Create a key if needed:
   ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
2. Install the public key on one login host:
   ssh-copy-id -i ~/.ssh/id_ed25519.pub <SLURM_USER>@<login-host>
3. Trust the host key:
   ssh-keyscan -H <login-host> >> ~/.ssh/known_hosts
4. Lock private-key permissions:
   chmod 600 ~/.ssh/id_ed25519
5. Verify it works without prompts:
   ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 <SLURM_USER>@<login-host> 'hostname'

After that, rerun with SSH_KEY_PATH=~/.ssh/id_ed25519.
```

## Result Path

Results default to:

```text
/lustre/fsw/portfolios/edgeai/users/<slurm_user>/results/<job_id>
```

The runner sets `TAO_API_RESULTS_DIR` to the parent results directory because
container code appends the job id when writing status and artifacts.
