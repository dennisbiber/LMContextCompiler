"""
Prompt utilities — template rendering and conversation history formatting.

The {{{variable_name}}} triple-brace syntax is preserved from the original
so existing prompt text files require zero changes.
"""

from pathlib import Path
from typing import List, Dict


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def render_template(path: str, variables: dict) -> str:
    """
    Read a prompt text file and substitute {{{variable_name}}} placeholders.

    All substitutions are done with str.replace so there are no conflicts
    with Python's own str.format() syntax.  Missing variables are left as-is
    so failures are obvious in the rendered output.
    """
    text = Path(path).read_text(encoding="utf-8")
    for key, value in variables.items():
        text = text.replace(f"{{{{{{{key}}}}}}}", str(value) if value is not None else "")
    return text


# ---------------------------------------------------------------------------
# Conversation history helpers
# ---------------------------------------------------------------------------

def format_history(history: List[Dict]) -> str:
    """
    Convert a list of {role: content} dicts into a plain-text transcript.

    Example output:
        CHARACTER_NAME: Hello there.
        USERNAME: Hi!
    """
    lines = []
    for entry in history:
        role = list(entry.keys())[0]
        text = list(entry.values())[0]
        lines.append(f"{role.upper()}: {text}")
    return "\n".join(lines)


def build_conversation_windows(
    history: List[Dict],
    recent_ratio: float = 0.5,
    max_full: int = 36,
):
    """
    Split history into a full window and a recent window.

    Returns (full_text, recent_text) as plain strings.
    """
    if not history:
        return "", ""

    full = history[-max_full:] if max_full else history
    recent_start = int(len(full) * (1 - recent_ratio))
    recent = full[recent_start:]
    older = full[:recent_start]

    return format_history(older), format_history(recent)
