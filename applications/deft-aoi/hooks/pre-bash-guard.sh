#!/usr/bin/env bash
# PreToolUse(Bash) hook: block destructive commands that could wipe DEFT loop state.

set -euo pipefail

WORKSPACE="${DEFT_WORKSPACE:-$PWD}"

STATE_FILE=$(find "$WORKSPACE" -name "deft_state.json" -not -path "*/node_modules/*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$STATE_FILE" ]; then
  exit 0
fi

RESULTS_DIR=$(dirname "$STATE_FILE")

# Read the command from stdin
COMMAND=$(python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('tool_input', {}).get('command', ''))
" 2>/dev/null || true)

if [ -z "$COMMAND" ]; then
  exit 0
fi

# Check for destructive patterns targeting the results directory
BLOCKED=$(python3 -c "
import re, sys

command = '''${COMMAND}'''
results_dir = '${RESULTS_DIR}'

dangers = []

# rm -rf on results dir or its parents
if re.search(r'rm\s+(-[rfR]+\s+)*' + re.escape(results_dir), command):
    dangers.append('rm targeting DEFT results directory')

# rm -rf on deft_state.json
if 'deft_state.json' in command and 'rm' in command:
    dangers.append('rm targeting deft_state.json')

# docker system prune while loop is active
if 'docker system prune' in command or 'docker container prune' in command:
    dangers.append('docker prune could remove containers needed by DEFT loop')

# Overwriting state file with redirect
if 'deft_state.json' in command and '>' in command and 'python3' not in command and 'jq' not in command:
    dangers.append('raw redirect into deft_state.json (use python3/jq to update safely)')

if dangers:
    print(' | '.join(dangers))
" 2>/dev/null || true)

if [ -z "$BLOCKED" ]; then
  exit 0
fi

python3 -c "
import json, sys
output = {
    'hookSpecificOutput': {
        'hookEventName': 'PreToolUse',
        'permissionDecision': 'deny',
        'permissionDecisionReason': 'BLOCKED: ${BLOCKED}. DEFT loop is active with state at ${RESULTS_DIR}/deft_state.json. Do not destroy loop state. If you need to clean up, target specific subdirectories, not the entire results tree.'
    }
}
json.dump(output, sys.stdout)
"
