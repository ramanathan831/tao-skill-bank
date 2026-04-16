#!/usr/bin/env bash
# Stop hook: if a DEFT loop is active and incomplete, tell Claude to keep going.

set -euo pipefail

WORKSPACE="${DEFT_WORKSPACE:-$PWD}"

STATE_FILE=$(find "$WORKSPACE" -name "deft_state.json" -not -path "*/node_modules/*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$STATE_FILE" ]; then
  exit 0
fi

# Check if loop is still in progress
INCOMPLETE=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
step = state.get('current_step', '')
max_iter = state.get('max_iterations', 3)
cur_iter = state.get('current_iteration', 0)
iterations = state.get('iterations', {})

# Check if we're done
if step == 'finished':
    exit(0)

# Check if the latest iteration is complete and KPI met
for key in sorted(iterations.keys(), reverse=True):
    it = iterations[key]
    if it.get('status') == 'complete' and it.get('kpi_met'):
        exit(0)

# Check if we've exhausted max iterations
completed = sum(1 for v in iterations.values() if v.get('status') == 'complete')
if completed > max_iter:
    exit(0)

# Check for unresolved errors — still block, but include error context
has_error = False
error_msg = ''
for key in sorted(iterations.keys(), reverse=True):
    it = iterations[key]
    err = it.get('last_error', {})
    if err and not err.get('resolved', True):
        has_error = True
        error_msg = f"UNRESOLVED ERROR in {key}/{err.get('step','?')}: {err.get('error','?')} (retry #{err.get('retry_count',0)})"
        break

# Still in progress
info = f'iter={cur_iter} step={step} max={max_iter}'
if has_error:
    info += f' | {error_msg}'
print(info)
" 2>/dev/null || true)

if [ -z "$INCOMPLETE" ]; then
  exit 0
fi

RESULTS_DIR=$(dirname "$STATE_FILE")

python3 -c "
import json, sys
output = {
    'decision': 'block',
    'reason': 'DEFT loop is still in progress (${INCOMPLETE}). Do not stop — diagnose any errors and continue to the next step. Re-read ${RESULTS_DIR}/deft_state.json if you lost track of where you are.'
}
json.dump(output, sys.stdout)
"
