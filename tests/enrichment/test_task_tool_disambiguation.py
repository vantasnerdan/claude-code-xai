"""Tests for Issue #34: TaskCreate vs Task tool disambiguation.

Verifies that the system preamble, tool_knowledge, and YAML definitions
correctly distinguish between:
- TaskCreate (tracking entry only, no work performed)
- Task (launches a real subagent that performs work)
- TaskUpdate/TaskGet/TaskList (tracking management only)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from enrichment.system_preamble import _PREAMBLE
from enrichment.behavioral.tool_knowledge import TOOL_KNOWLEDGE


# --- The five orchestration tools covered by this fix ---
ORCHESTRATION_TOOLS = ["TaskCreate", "Task", "TaskUpdate", "TaskGet", "TaskList"]
TRACKING_ONLY_TOOLS = ["TaskCreate", "TaskUpdate", "TaskGet", "TaskList"]


class TestSystemPreambleSection7:
    """System preamble must contain Section 7: Orchestration Tools."""

    def test_section_7_exists(self) -> None:
        """Preamble contains a Section 7 heading."""
        assert "## 7." in _PREAMBLE

    def test_section_7_title(self) -> None:
        """Section 7 is titled 'Orchestration Tools'."""
        assert "Orchestration Tools" in _PREAMBLE

    def test_taskcreate_tracking_only(self) -> None:
        """Preamble explicitly states TaskCreate is tracking only."""
        assert "TaskCreate" in _PREAMBLE
        assert "TRACKING ENTRY ONLY" in _PREAMBLE

    def test_task_launches_subagent(self) -> None:
        """Preamble explicitly states Task launches a real subagent."""
        assert "LAUNCHES A REAL SUBAGENT" in _PREAMBLE

    def test_planning_is_not_doing(self) -> None:
        """Preamble warns that planning is not doing."""
        assert "Planning is NOT doing" in _PREAMBLE

    def test_tracking_entry_does_not_launch(self) -> None:
        """Preamble states creating a tracking entry does NOT launch work."""
        assert "Creating a tracking entry does NOT launch work" in _PREAMBLE

    def test_structured_prompt_template(self) -> None:
        """Preamble includes structured prompt template components."""
        assert "Task scope" in _PREAMBLE
        assert "Acceptance criteria" in _PREAMBLE
        assert "Constraints" in _PREAMBLE

    def test_run_in_background(self) -> None:
        """Preamble mentions run_in_background for non-blocking execution."""
        assert "run_in_background" in _PREAMBLE

    def test_workflow_steps(self) -> None:
        """Preamble describes the planning-then-execution workflow."""
        assert "TaskCreate to plan and track" in _PREAMBLE
        assert "Task tool to execute" in _PREAMBLE

    def test_tracking_management_tools(self) -> None:
        """Preamble mentions TaskUpdate/TaskGet/TaskList as tracking only."""
        assert "TaskUpdate" in _PREAMBLE
        assert "TaskGet" in _PREAMBLE
        assert "TaskList" in _PREAMBLE
        assert "tracking management only" in _PREAMBLE


class TestToolKnowledgeOrchestration:
    """tool_knowledge.py must have WHAT/WHY/WHEN for all 5 orchestration tools."""

    @pytest.mark.parametrize("tool_name", ORCHESTRATION_TOOLS)
    def test_tool_has_what(self, tool_name: str) -> None:
        """Each orchestration tool has a 'what' entry."""
        assert tool_name in TOOL_KNOWLEDGE
        assert "what" in TOOL_KNOWLEDGE[tool_name]
        assert isinstance(TOOL_KNOWLEDGE[tool_name]["what"], str)
        assert len(TOOL_KNOWLEDGE[tool_name]["what"]) > 20

    @pytest.mark.parametrize("tool_name", ORCHESTRATION_TOOLS)
    def test_tool_has_why(self, tool_name: str) -> None:
        """Each orchestration tool has a 'why' entry with problem_context and failure_modes."""
        assert "why" in TOOL_KNOWLEDGE[tool_name]
        why = TOOL_KNOWLEDGE[tool_name]["why"]
        assert "problem_context" in why
        assert "failure_modes" in why
        assert isinstance(why["failure_modes"], list)
        assert len(why["failure_modes"]) > 0

    @pytest.mark.parametrize("tool_name", ORCHESTRATION_TOOLS)
    def test_tool_has_when(self, tool_name: str) -> None:
        """Each orchestration tool has a 'when' entry with prerequisites and sequencing."""
        assert "when" in TOOL_KNOWLEDGE[tool_name]
        when = TOOL_KNOWLEDGE[tool_name]["when"]
        assert "prerequisites" in when
        assert "sequencing" in when
        assert isinstance(when["sequencing"], str)

    def test_taskcreate_what_says_tracking_only(self) -> None:
        """TaskCreate WHAT explicitly says no work is performed."""
        what = TOOL_KNOWLEDGE["TaskCreate"]["what"]
        assert "TRACKING ENTRY" in what
        assert "No work is performed" in what
        assert "No subagent is launched" in what

    def test_task_what_says_launches_subagent(self) -> None:
        """Task WHAT explicitly says it launches a real subagent."""
        what = TOOL_KNOWLEDGE["Task"]["what"]
        assert "REAL SUBAGENT" in what
        assert "autonomously performs work" in what

    def test_taskcreate_why_warns_of_confusion(self) -> None:
        """TaskCreate WHY failure_modes warns about Task confusion."""
        modes = TOOL_KNOWLEDGE["TaskCreate"]["why"]["failure_modes"]
        confusion_warnings = [m for m in modes if "Task tool" in m or "subagent" in m]
        assert len(confusion_warnings) > 0

    def test_task_why_warns_of_confusion(self) -> None:
        """Task WHY failure_modes warns about TaskCreate confusion."""
        modes = TOOL_KNOWLEDGE["Task"]["why"]["failure_modes"]
        confusion_warnings = [m for m in modes if "TaskCreate" in m]
        assert len(confusion_warnings) > 0

    def test_task_when_includes_prompt_template(self) -> None:
        """Task WHEN sequencing mentions structured prompt components."""
        sequencing = TOOL_KNOWLEDGE["Task"]["when"]["sequencing"]
        assert "task scope" in sequencing
        assert "acceptance criteria" in sequencing
        assert "constraints" in sequencing

    @pytest.mark.parametrize("tool_name", TRACKING_ONLY_TOOLS)
    def test_tracking_tools_what_says_no_work(self, tool_name: str) -> None:
        """All tracking-only tools explicitly state no work is performed."""
        what = TOOL_KNOWLEDGE[tool_name]["what"].lower()
        assert "no work" in what or "does not perform" in what or "tracking entry" in what


class TestYamlOrchestrationTools:
    """YAML behavioral files must have entries for all 5 orchestration tools."""

    @pytest.fixture
    def yaml_dir(self) -> Path:
        """Path to the behavioral YAML directory."""
        return Path(__file__).resolve().parent.parent.parent / "structure" / "behavioral"

    @pytest.fixture
    def what_yaml(self, yaml_dir: Path) -> dict[str, Any]:
        """Parsed what.yaml."""
        with open(yaml_dir / "what.yaml") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def why_yaml(self, yaml_dir: Path) -> dict[str, Any]:
        """Parsed why.yaml."""
        with open(yaml_dir / "why.yaml") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def when_yaml(self, yaml_dir: Path) -> dict[str, Any]:
        """Parsed when.yaml."""
        with open(yaml_dir / "when.yaml") as f:
            return yaml.safe_load(f)

    @pytest.mark.parametrize("tool_name", ORCHESTRATION_TOOLS)
    def test_what_yaml_has_tool(self, what_yaml: dict, tool_name: str) -> None:
        """what.yaml has an entry for each orchestration tool."""
        assert tool_name in what_yaml["tools"], f"{tool_name} missing from what.yaml"

    @pytest.mark.parametrize("tool_name", ORCHESTRATION_TOOLS)
    def test_why_yaml_has_tool(self, why_yaml: dict, tool_name: str) -> None:
        """why.yaml has an entry for each orchestration tool."""
        assert tool_name in why_yaml["tools"], f"{tool_name} missing from why.yaml"

    @pytest.mark.parametrize("tool_name", ORCHESTRATION_TOOLS)
    def test_when_yaml_has_tool(self, when_yaml: dict, tool_name: str) -> None:
        """when.yaml has an entry for each orchestration tool."""
        assert tool_name in when_yaml["tools"], f"{tool_name} missing from when.yaml"

    def test_what_yaml_taskcreate_tracking_only(self, what_yaml: dict) -> None:
        """what.yaml TaskCreate description emphasizes tracking only."""
        text = what_yaml["tools"]["TaskCreate"]
        assert "TRACKING ENTRY" in text

    def test_what_yaml_task_launches_subagent(self, what_yaml: dict) -> None:
        """what.yaml Task description emphasizes real subagent."""
        text = what_yaml["tools"]["Task"]
        assert "REAL SUBAGENT" in text

    def test_why_yaml_taskcreate_warns_confusion(self, why_yaml: dict) -> None:
        """why.yaml TaskCreate failure_modes includes confusion warning."""
        modes = why_yaml["tools"]["TaskCreate"]["failure_modes"]
        assert any("Task tool" in m or "subagent" in m for m in modes)

    def test_why_yaml_task_warns_confusion(self, why_yaml: dict) -> None:
        """why.yaml Task failure_modes includes TaskCreate confusion warning."""
        modes = why_yaml["tools"]["Task"]["failure_modes"]
        assert any("TaskCreate" in m for m in modes)

    def test_when_yaml_taskcreate_use_before_task(self, when_yaml: dict) -> None:
        """when.yaml TaskCreate lists Task in use_before."""
        use_before = when_yaml["tools"]["TaskCreate"]["use_before"]
        assert "Task" in use_before

    @pytest.mark.parametrize("tool_name", ORCHESTRATION_TOOLS)
    def test_when_yaml_has_sequencing(self, when_yaml: dict, tool_name: str) -> None:
        """when.yaml has a sequencing string for each orchestration tool."""
        tool = when_yaml["tools"][tool_name]
        assert "sequencing" in tool
        assert isinstance(tool["sequencing"], str)
        assert len(tool["sequencing"]) > 10


class TestYamlTypeSafety:
    """Guard against YAML parsing bugs where colon-space in strings produces dicts."""

    @pytest.fixture
    def yaml_dir(self) -> Path:
        """Path to the behavioral YAML directory."""
        return Path(__file__).resolve().parent.parent.parent / "structure" / "behavioral"

    @pytest.fixture
    def why_yaml(self, yaml_dir: Path) -> dict[str, Any]:
        """Parsed why.yaml."""
        with open(yaml_dir / "why.yaml") as f:
            return yaml.safe_load(f)

    def test_all_failure_modes_are_strings(self, why_yaml: dict) -> None:
        """Every failure_mode entry in why.yaml must be a string, not a dict.

        YAML parses unquoted 'key: value' as a mapping. If a failure_mode
        contains a colon-space (e.g. 'run_in_background: true'), it must
        be quoted to remain a string.
        """
        tools = why_yaml["tools"]
        for tool_name, tool_data in tools.items():
            if "failure_modes" not in tool_data:
                continue
            for i, mode in enumerate(tool_data["failure_modes"]):
                assert isinstance(mode, str), (
                    f"{tool_name} failure_modes[{i}] is {type(mode).__name__}, "
                    f"not str. Likely unquoted colon-space in YAML. "
                    f"Value: {mode!r}"
                )

    @pytest.mark.parametrize("tool_name", ORCHESTRATION_TOOLS)
    def test_orchestration_failure_modes_are_strings(
        self, why_yaml: dict, tool_name: str
    ) -> None:
        """Each orchestration tool's failure_modes are all strings."""
        modes = why_yaml["tools"][tool_name]["failure_modes"]
        for i, mode in enumerate(modes):
            assert isinstance(mode, str), (
                f"{tool_name} failure_modes[{i}] is {type(mode).__name__}, "
                f"not str. Value: {mode!r}"
            )


class TestPreambleYamlSection7:
    """The preamble behavioral.yaml must also contain Section 7."""

    @pytest.fixture
    def preamble_yaml(self) -> dict[str, Any]:
        """Parsed behavioral preamble YAML."""
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "structure"
            / "preamble"
            / "behavioral.yaml"
        )
        with open(path) as f:
            return yaml.safe_load(f)

    def test_preamble_yaml_contains_section_7(self, preamble_yaml: dict) -> None:
        """Preamble YAML text contains Section 7."""
        text = preamble_yaml["text"]
        assert "## 7." in text

    def test_preamble_yaml_orchestration_tools(self, preamble_yaml: dict) -> None:
        """Preamble YAML text mentions orchestration tools."""
        text = preamble_yaml["text"]
        assert "Orchestration Tools" in text
        assert "TaskCreate" in text
        assert "TRACKING ENTRY ONLY" in text
        assert "LAUNCHES A REAL SUBAGENT" in text
