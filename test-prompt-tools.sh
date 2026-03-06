#!/bin/bash
# Test: Can Grok 4.20 multi-agent do prompt-based tool calling?
# Run: XAI_API_KEY=your-key bash test-prompt-tools.sh

set -euo pipefail

API_KEY="${XAI_API_KEY:?Set XAI_API_KEY}"

curl -s https://api.x.ai/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "$(cat <<'PAYLOAD'
{
  "model": "grok-4.20-multi-agent-experimental-beta-0304",
  "store": false,
  "input": [
    {
      "role": "system",
      "content": "You are a coding assistant with access to tools. To use a tool, output ONLY a fenced JSON block in this exact format:\n\n```json\n{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value\"}}\n```\n\nDo NOT add any text before or after the JSON block when making a tool call. No commentary, no explanation — just the fenced JSON.\n\nAfter tool results are provided, continue reasoning or make another tool call.\n\nAvailable tools:\n\n1. Read\n   Description: Read a file from the filesystem.\n   Parameters:\n   - file_path (string, required): Absolute path to the file\n   - offset (number, optional): Line number to start from\n   - limit (number, optional): Number of lines to read\n\n2. Bash  \n   Description: Execute a bash command and return output.\n   Parameters:\n   - command (string, required): The command to execute\n\n3. Write\n   Description: Write content to a file (creates or overwrites).\n   Parameters:\n   - file_path (string, required): Absolute path to the file\n   - content (string, required): Content to write\n\nExample of correct tool usage:\nUser: What files are in /tmp?\nAssistant:\n```json\n{\"name\": \"Bash\", \"parameters\": {\"command\": \"ls /tmp\"}}\n```\n\nExample of INCORRECT usage (do NOT do this):\nAssistant: Let me check that for you.\n```json\n{\"name\": \"Bash\", \"parameters\": {\"command\": \"ls /tmp\"}}\n```\nI'll look at those files now.\n\nThe incorrect example has text before and after the JSON block. Only output the JSON block."
    },
    {
      "role": "user",
      "content": "Read the file at /opt/sage/sage-state/CLAUDE.md and tell me what the first line says."
    }
  ]
}
PAYLOAD
)" | python3 -m json.tool 2>/dev/null || echo "(raw output above)"
