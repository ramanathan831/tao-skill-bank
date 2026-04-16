#!/usr/bin/env bash
# PostToolUse(Bash) hook: after bash commands during a DEFT loop, check for
# Docker failures and missing expected outputs.

set -euo pipefail

WORKSPACE="${DEFT_WORKSPACE:-$PWD}"

STATE_FILE=$(find "$WORKSPACE" -name "deft_state.json" -not -path "*/node_modules/*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$STATE_FILE" ]; then
  exit 0
fi

# Read the tool input from stdin
INPUT=$(cat)

COMMAND=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('tool_input', {}).get('command', ''))
" 2>/dev/null || true)

# Only check docker-related commands
if ! echo "$COMMAND" | grep -qE '(docker run|docker exec|nvidia-smi)'; then
  exit 0
fi

# Check tool response for failure signals
WARNINGS=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
response = str(data.get('tool_response', ''))
warnings = []

# Docker/CUDA failure patterns
patterns = [
    ('out of memory', 'GPU OOM — reduce batch size or free GPU memory'),
    ('CUDA error', 'CUDA runtime error — check nvidia-smi, GPU may be busy'),
    ('OCI runtime create failed', 'Docker container failed to start — check image/GPU access'),
    ('No such container', 'Docker container not found — may have exited prematurely'),
    ('Killed', 'Process killed (likely OOM) — reduce batch size or check system memory'),
    ('Error response from daemon', 'Docker daemon error — check disk space and Docker status'),
    ('NCCL error', 'Multi-GPU communication error — try single GPU or check NCCL config'),
    ('RuntimeError: DataLoader worker', 'DataLoader crash — reduce num_workers or check shared memory'),
    ('Permission denied', 'Permission error — may need sudo chown on output directory'),
    ('No space left on device', 'Disk full — free space before continuing'),
]

for pattern, msg in patterns:
    if pattern.lower() in response.lower():
        warnings.append(msg)

if warnings:
    print('\n'.join(warnings))
" 2>/dev/null || true)

RESULTS_DIR=$(dirname "$STATE_FILE")

# Detect what kind of step just ran (for checkpoint reminders)
STEP_TYPE=$(echo "$COMMAND" | python3 -c "
import sys
cmd = sys.stdin.read()
if 'tao-toolkit' in cmd or 'classification' in cmd and 'train' in cmd:
    print('training')
elif 'inference' in cmd or 'tao-toolkit' in cmd and 'evaluate' in cmd:
    print('inference')
elif 'anomalygen' in cmd:
    print('anomalygen')
elif 'pcb-aoi-ov-sdg' in cmd or 'sdg_pipeline' in cmd:
    print('sdg')
elif 'embed' in cmd or 'mining' in cmd:
    print('mining')
else:
    print('')
" 2>/dev/null || true)

# If failures detected, record to state file and warn
if [ -n "$WARNINGS" ]; then
  # Write failure into deft_state.json
  python3 -c "
import json
from datetime import datetime, timezone

with open('${STATE_FILE}', 'r') as f:
    state = json.load(f)

cur_iter = state.get('current_iteration', 0)
iter_key = 'baseline' if cur_iter == 0 else f'iter{cur_iter}'

if iter_key not in state.get('iterations', {}):
    state.setdefault('iterations', {})[iter_key] = {}

it = state['iterations'][iter_key]

# Initialize or increment failure tracking
prev = it.get('last_error', {})
same_step = prev.get('step', '') == '${STEP_TYPE:-unknown}'
retry_count = (prev.get('retry_count', 0) + 1) if same_step else 1

it['last_error'] = {
    'step': '${STEP_TYPE:-unknown}',
    'error': '''${WARNINGS}'''.strip(),
    'command': '''${COMMAND}'''[:200],
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'retry_count': retry_count,
    'resolved': False
}
it['status'] = 'error'

with open('${STATE_FILE}', 'w') as f:
    json.dump(state, f, indent=2)
" 2>/dev/null || true

  # Notify Claude
  python3 -c "
import json, sys
output = {
    'hookSpecificOutput': {
        'hookEventName': 'PostToolUse',
        'additionalContext': '''FAILURE recorded in deft_state.json.

Error: ${WARNINGS}
Step: ${STEP_TYPE:-unknown}

The failure has been saved to the state file with retry count.
Diagnose the issue and retry. When the step succeeds, the checkpoint
reminder will clear the error (set resolved: true).

If this is the 3rd+ retry for the same step, consider:
- Reducing batch size (OOM)
- Freeing GPU memory (nvidia-smi to check)
- Checking disk space (df -h)
- Asking the user for help

Do NOT proceed to the next step until this is resolved.'''
    }
}
json.dump(output, sys.stdout)
"
  exit 0
fi

# If no failures and it was a significant step, clear error + remind to checkpoint
if [ -n "$STEP_TYPE" ]; then
  # Clear any previous error for this step
  python3 -c "
import json

with open('${STATE_FILE}', 'r') as f:
    state = json.load(f)

cur_iter = state.get('current_iteration', 0)
iter_key = 'baseline' if cur_iter == 0 else f'iter{cur_iter}'
it = state.get('iterations', {}).get(iter_key, {})

prev_err = it.get('last_error', {})
if prev_err and not prev_err.get('resolved', True):
    it['last_error']['resolved'] = True
    if it.get('status') == 'error':
        it['status'] = 'in_progress'
    with open('${STATE_FILE}', 'w') as f:
        json.dump(state, f, indent=2)
" 2>/dev/null || true

  python3 -c "
import json, sys
output = {
    'hookSpecificOutput': {
        'hookEventName': 'PostToolUse',
        'additionalContext': '''DEFT checkpoint reminder: ${STEP_TYPE} command completed successfully.

Update ${RESULTS_DIR}/deft_state.json now:
- Set current_step to reflect this completion
- Record any outputs (checkpoint path, image counts, metrics)
- Write iteration summary if this was the last step of an iteration
- Any previous error for this step has been marked resolved.

Do this before starting the next step.'''
    }
}
json.dump(output, sys.stdout)
"
fi
