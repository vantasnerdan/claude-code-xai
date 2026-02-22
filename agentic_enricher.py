def enrich_tools_for_grok(tools: list) -> list:
    for tool in tools:
        if "description" not in tool:
            tool["description"] = ""
        # Pattern 6 + 14 + 7 + 15
        tool["description"] += ("\n\nAGENTIC STANDARD (Gold): "
                                "Follow inputSchema exactly. "
                                "Return _links and suggestion on any error. "
                                "Use canonical names only. "
                                "On failure follow anti-pattern registry.")
        # Example anti-patterns from your repo
        if tool.get("name") == "bash":
            tool["description"] += "\nANTI-PATTERN: Never rm -rf without explicit confirmation."
        # Pattern 16 versioning
        tool["inputSchema"] = tool.get("input_schema", {})  # normalize
        tool.setdefault("version", "1.0")
    return tools
