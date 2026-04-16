#!/usr/bin/env bash
# SessionStart hook: detect an in-progress DEFT loop and prompt Claude to resume.

set -euo pipefail

WORKSPACE="${DEFT_WORKSPACE:-$PWD}"

STATE_FILE=$(find "$WORKSPACE" -name "deft_state.json" -not -path "*/node_modules/*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$STATE_FILE" ]; then
  exit 0
fi

RESULTS_DIR=$(dirname "$STATE_FILE")

# Check if loop is incomplete
STATUS=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
step = state.get('current_step', '')
if step == 'finished':
    exit(0)
cur_iter = state.get('current_iteration', 0)
kpi = state.get('kpi_target', '?')
iterations = state.get('iterations', {})
# Check if latest iteration has kpi_met
for key in sorted(iterations.keys(), reverse=True):
    if iterations[key].get('kpi_met'):
        exit(0)
print(f'iter={cur_iter}|step={step}|kpi={kpi}')
" 2>/dev/null || true)

if [ -z "$STATUS" ]; then
  exit 0
fi

ITER=$(echo "$STATUS" | cut -d'|' -f1 | cut -d'=' -f2)
STEP=$(echo "$STATUS" | cut -d'|' -f2 | cut -d'=' -f2)
KPI=$(echo "$STATUS" | cut -d'|' -f3 | cut -d'=' -f2-)

# Gather summaries
SUMMARIES=""
for f in "$RESULTS_DIR"/*_summary.md; do
  [ -f "$f" ] && SUMMARIES+="$(cat "$f")\n\n---\n\n"
done

python3 -c "
import json, sys
context = '''DEFT LOOP RESUME — An in-progress DEFT loop was detected.

State file: ${RESULTS_DIR}/deft_state.json
Current position: iteration ${ITER}, step: ${STEP}
KPI target: ${KPI}

Read the state file and resume from where you left off. Do NOT re-collect
inputs or re-run completed steps. Jump directly to the incomplete step.

${SUMMARIES:+Previous iteration summaries:\n${SUMMARIES}}
Re-read the latest RCA report (path in state file) before making strategy decisions.'''

output = {
    'hookSpecificOutput': {
        'hookEventName': 'SessionStart',
        'additionalContext': context
    }
}
json.dump(output, sys.stdout)
"
