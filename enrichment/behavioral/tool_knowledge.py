"""Claude Code tool knowledge base.

This module contains the behavioral knowledge that Claude models learn via
RL training. By making it explicit and structured, any model can perform
like a trained one.

Source: Claude Code system prompt tool descriptions + observed patterns.

Each tool gets WHAT/WHY/WHEN dimensions:
  WHAT — What the tool does (enhanced beyond the raw description)
  WHY  — Why to use it (problem context, failure modes)
  WHEN — When to use it (prerequisites, alternatives, sequencing)
"""
from typing import Any

TOOL_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "Read": {
        "what": (
            "Reads file contents from the local filesystem. Supports text files, "
            "images (PNG, JPG — rendered visually), PDFs (up to 20 pages per request), "
            "and Jupyter notebooks (all cells with outputs). Returns content with "
            "line numbers (cat -n format). Lines longer than 2000 characters are truncated."
        ),
        "why": {
            "problem_context": (
                "You must understand existing code before modifying it. Reading "
                "prevents blind edits that break things. The line numbers in output "
                "help you construct exact old_string matches for Edit."
            ),
            "failure_modes": [
                "Editing a file without reading it first will likely produce incorrect old_string matches",
                "Using Bash cat/head/tail instead of Read loses line number context",
                "Reading a large PDF without the pages parameter will fail — use pages for PDFs over 10 pages",
                "Passing a relative path instead of an absolute path will fail",
            ],
        },
        "when": {
            "prerequisites": [],
            "use_before": ["Edit", "Write"],
            "use_instead_of": ["Bash cat", "Bash head", "Bash tail"],
            "sequencing": (
                "Use after Glob/Grep to find files, before Edit to modify them. "
                "For large files, use offset and limit parameters to read in chunks."
            ),
        },
    },
    "Edit": {
        "what": (
            "Performs exact string replacements in files. Surgical edits that "
            "preserve surrounding code. The old_string must be unique in the file "
            "or the edit will fail. Use replace_all=True for renaming across the file."
        ),
        "why": {
            "problem_context": (
                "Surgical edits preserve surrounding code. Full rewrites via Write "
                "risk losing context and introducing bugs. The exact-match requirement "
                "prevents accidental edits to the wrong location."
            ),
            "failure_modes": [
                "Using Edit without Read first — the old_string won't match if you haven't seen the actual file content",
                "Using Bash sed/awk instead of Edit loses the safety of exact-match replacement",
                "Non-unique old_string causes the edit to fail — provide more surrounding context to disambiguate",
                "Including line number prefixes from Read output in old_string — those are display artifacts, not file content",
            ],
        },
        "when": {
            "prerequisites": ["Read"],
            "use_before": [],
            "use_instead_of": ["Bash sed", "Bash awk"],
            "sequencing": (
                "Always Read the file first. Use Edit for targeted changes, Write "
                "only for new files or complete rewrites."
            ),
        },
    },
    "Write": {
        "what": (
            "Creates or completely overwrites a file on the local filesystem. "
            "If the file exists, you MUST Read it first. Prefer Edit for existing files."
        ),
        "why": {
            "problem_context": (
                "Use for creating new files. For existing files, prefer Edit to "
                "preserve surrounding code. Write overwrites the entire file, so "
                "anything not included in your content is lost."
            ),
            "failure_modes": [
                "Overwriting an existing file without reading it first loses all content not in your write",
                "Creating files unnecessarily leads to file bloat — always prefer editing existing files",
                "Writing documentation files (*.md, README) proactively when not asked creates unwanted artifacts",
            ],
        },
        "when": {
            "prerequisites": ["Read (if file exists)"],
            "use_before": [],
            "use_instead_of": ["Bash echo", "Bash cat heredoc"],
            "sequencing": (
                "Check if file exists first. If yes, prefer Edit. If no, Write is correct. "
                "Never proactively create documentation files unless explicitly requested."
            ),
        },
    },
    "Bash": {
        "what": (
            "Executes bash commands with optional timeout (up to 10 minutes). "
            "Working directory persists between calls; shell state does not. "
            "For git, npm, docker, and system operations — NOT for file operations."
        ),
        "why": {
            "problem_context": (
                "Terminal operations that require shell execution. Dedicated tools "
                "exist for file operations and should be used instead for safety "
                "and integration with the permission system."
            ),
            "failure_modes": [
                "Using Bash for file reads (cat) instead of Read tool",
                "Using Bash for file search (find, grep) instead of Glob/Grep tools",
                "Using Bash for file edits (sed, awk) instead of Edit tool",
                "Using Bash for file writes (echo >) instead of Write tool",
                "Not quoting file paths with spaces causes command failure",
                "Using interactive flags (-i) with git commands fails — no interactive input supported",
            ],
        },
        "when": {
            "prerequisites": [],
            "use_before": [],
            "use_instead_of": [],
            "do_not_use_for": [
                "Reading files (use Read)",
                "Searching files (use Grep/Glob)",
                "Editing files (use Edit)",
                "Writing files (use Write)",
            ],
            "sequencing": (
                "Reserve for git operations, package management, running tests, "
                "docker commands, and other terminal-specific tasks. Chain dependent "
                "commands with && not newlines."
            ),
        },
    },
    "Grep": {
        "what": (
            "Searches file contents using regex patterns. Built on ripgrep for speed. "
            "Supports output modes: content (matching lines), files_with_matches (paths only), "
            "count (match counts). Supports context lines (-A, -B, -C) and glob filtering."
        ),
        "why": {
            "problem_context": (
                "Find code patterns, function definitions, imports, and usages across "
                "the entire codebase efficiently. Integrated with the permission system."
            ),
            "failure_modes": [
                "Using Bash grep/rg instead of the Grep tool loses integration with the permission system",
                "Searching without glob filter in large repos returns too many results",
                "Literal braces in patterns need escaping (use interface\\{\\} to find interface{} in Go)",
            ],
        },
        "when": {
            "prerequisites": [],
            "use_before": ["Read"],
            "use_instead_of": ["Bash grep", "Bash rg"],
            "sequencing": (
                "Use to find relevant files/lines, then Read for full context, "
                "then Edit to modify. Combine with Glob when you need both name "
                "and content matching."
            ),
        },
    },
    "Glob": {
        "what": (
            "Fast file pattern matching. Find files by name patterns like "
            "'**/*.py' or 'src/**/*.ts'. Returns matching paths sorted by "
            "modification time."
        ),
        "why": {
            "problem_context": (
                "Locate files by name or extension before reading or searching "
                "their contents. Faster and more integrated than Bash find."
            ),
            "failure_modes": [
                "Using Bash find instead of Glob loses integration and is slower",
                "Using Bash ls for file discovery misses nested files",
            ],
        },
        "when": {
            "prerequisites": [],
            "use_before": ["Read", "Grep"],
            "use_instead_of": ["Bash find", "Bash ls"],
            "sequencing": (
                "Use first to locate files by name, then Grep to search contents, "
                "then Read for full file context. Multiple Glob calls can run in "
                "parallel for different patterns."
            ),
        },
    },
    "WebFetch": {
        "what": (
            "Fetches content from a URL and processes it with an AI model. "
            "Converts HTML to markdown. Includes a 15-minute cache. "
            "WILL FAIL for authenticated or private URLs."
        ),
        "why": {
            "problem_context": (
                "Retrieve and analyze public web content. Not suitable for "
                "authenticated services — use specialized tools for GitHub, "
                "Jira, Confluence, etc."
            ),
            "failure_modes": [
                "Fetching authenticated URLs (Google Docs, private GitHub) will fail",
                "For GitHub URLs, prefer gh CLI via Bash instead",
                "HTTP URLs are automatically upgraded to HTTPS",
            ],
        },
        "when": {
            "prerequisites": [],
            "use_before": [],
            "use_instead_of": [],
            "do_not_use_for": [
                "GitHub operations (use gh CLI via Bash)",
                "Authenticated services (use specialized tools)",
            ],
            "sequencing": (
                "Check if a specialized tool exists first (e.g., gh for GitHub). "
                "Only use WebFetch for public, unauthenticated URLs."
            ),
        },
    },
    "WebSearch": {
        "what": (
            "Searches the web and returns results with links. Provides "
            "up-to-date information beyond the model's knowledge cutoff. "
            "Supports domain filtering."
        ),
        "why": {
            "problem_context": (
                "Access current information for recent events, documentation, "
                "or data that may have changed since the model's training cutoff."
            ),
            "failure_modes": [
                "Forgetting to include Sources section with URLs in the response",
                "Using the wrong year in search queries — always use the current year",
            ],
        },
        "when": {
            "prerequisites": [],
            "use_before": ["WebFetch"],
            "use_instead_of": [],
            "sequencing": (
                "Search first to find relevant URLs, then WebFetch specific pages "
                "for detailed content. Always include Sources section in response."
            ),
        },
    },
    "NotebookEdit": {
        "what": (
            "Replaces, inserts, or deletes cells in Jupyter notebooks (.ipynb). "
            "Cell numbering is 0-indexed. Supports code and markdown cell types."
        ),
        "why": {
            "problem_context": (
                "Edit Jupyter notebooks cell-by-cell rather than rewriting the "
                "entire JSON structure. Preserves notebook metadata and outputs."
            ),
            "failure_modes": [
                "Using Write to modify .ipynb files destroys notebook metadata and cell outputs",
                "Forgetting that cell_number is 0-indexed",
            ],
        },
        "when": {
            "prerequisites": ["Read (to see current notebook state)"],
            "use_before": [],
            "use_instead_of": ["Write (for .ipynb files)", "Edit (for .ipynb files)"],
            "sequencing": (
                "Read the notebook first to understand cell structure, then use "
                "NotebookEdit for targeted cell changes."
            ),
        },
    },
    # --- Orchestration Tools ---
    # These tools manage task tracking and subagent execution.
    # Critical: TaskCreate (tracking) and Task (execution) are frequently confused.
    "TaskCreate": {
        "what": (
            "Creates a TRACKING ENTRY in the task list. No work is performed. "
            "No subagent is launched. No API calls are made. It is a checklist "
            "item for organizing and planning what needs to be done."
        ),
        "why": {
            "problem_context": (
                "Task planning and task execution are separate steps. TaskCreate "
                "only records intent — it does not cause any action. Use it to "
                "organize a work plan before launching subagents with the Task tool."
            ),
            "failure_modes": [
                "Confusing TaskCreate with the Task tool — TaskCreate does NOT launch a subagent or perform work",
                "Believing that creating a task entry means work is happening in the background — it is not",
                "Using TaskCreate when you need actual execution — use the Task tool instead",
            ],
        },
        "when": {
            "prerequisites": [],
            "use_before": ["Task"],
            "use_instead_of": [],
            "sequencing": (
                "Use TaskCreate first to plan and organize work items. Then use "
                "the Task tool to actually execute each item. TaskCreate without "
                "a subsequent Task call means no work gets done."
            ),
        },
    },
    "Task": {
        "what": (
            "Launches a REAL SUBAGENT that autonomously performs work. The subagent "
            "reads files, makes API calls, writes code, and posts comments. Use "
            "run_in_background: true for non-blocking execution."
        ),
        "why": {
            "problem_context": (
                "This is the execution tool — the only way to delegate actual work "
                "to a subagent. Unlike TaskCreate (which only creates a tracking entry), "
                "the Task tool starts an autonomous agent that performs real operations."
            ),
            "failure_modes": [
                "Using TaskCreate instead of Task — TaskCreate only creates a tracking entry, no work happens",
                "Vague prompts produce vague results — always include scope, context, acceptance criteria, and constraints",
                "Forgetting run_in_background: true causes the orchestrator to block until the subagent finishes",
            ],
        },
        "when": {
            "prerequisites": ["TaskCreate (optional, for tracking)"],
            "use_before": ["TaskUpdate (to record completion)"],
            "use_instead_of": [],
            "sequencing": (
                "Structure the prompt with: task scope (one clear deliverable), "
                "context (file paths, prior decisions), acceptance criteria (how to "
                "verify completion), constraints (boundaries, what NOT to do), and "
                "reminders (skills to read, conventions to follow). Launch independent "
                "tasks in parallel with run_in_background: true."
            ),
        },
    },
    "TaskUpdate": {
        "what": (
            "Updates the status or metadata of an existing tracking entry. Does not "
            "perform any work. Does not launch a subagent."
        ),
        "why": {
            "problem_context": (
                "Tracking management — mark tasks as complete, update status, or "
                "add notes. This is bookkeeping, not execution."
            ),
            "failure_modes": [
                "Expecting TaskUpdate to trigger work — it only updates the tracking record",
            ],
        },
        "when": {
            "prerequisites": ["TaskCreate"],
            "use_before": [],
            "use_instead_of": [],
            "sequencing": (
                "Use after a Task subagent completes to record the outcome. "
                "Or use to update status during long-running work."
            ),
        },
    },
    "TaskGet": {
        "what": (
            "Reads a single tracking entry by ID. Returns task metadata, status, "
            "and description. Does not perform any work."
        ),
        "why": {
            "problem_context": (
                "Check the current state of a specific task. Useful for verifying "
                "whether a task has been completed or reviewing its details."
            ),
            "failure_modes": [
                "Expecting TaskGet to execute work — it only reads the tracking record",
            ],
        },
        "when": {
            "prerequisites": ["TaskCreate"],
            "use_before": ["TaskUpdate"],
            "use_instead_of": [],
            "sequencing": (
                "Use to check task status before deciding next steps. "
                "Combine with TaskList for a full picture of all tracked work."
            ),
        },
    },
    "TaskList": {
        "what": (
            "Lists all tracking entries with their status and metadata. Returns "
            "the full task list. Does not perform any work."
        ),
        "why": {
            "problem_context": (
                "Overview of all planned and in-progress work. Useful for "
                "identifying what has been completed, what is pending, and what "
                "is blocked."
            ),
            "failure_modes": [
                "Expecting TaskList to execute tasks — it only lists tracking records",
            ],
        },
        "when": {
            "prerequisites": [],
            "use_before": ["Task", "TaskUpdate"],
            "use_instead_of": [],
            "sequencing": (
                "Use at session start to review outstanding work, or after "
                "completing a batch of tasks to verify all items are done."
            ),
        },
    },
}
