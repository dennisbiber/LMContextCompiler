"""
Prompt renderers — SysPrompt and UserPrompt.

Each renderer is initialised with the full prompts dict from the config.
Calling .render(prompt_name, variables) returns the fully substituted string
ready to send to the LLM.

UserPrompt has two rendering modes:

  1. Template mode (existing behavior, unchanged):
     If the prompt config declares a "user_prompt" with a "path" key, the
     template file is loaded and {{{variable}}} substitutions are performed
     as before. Existing prompt files require zero changes.

  2. Auto-generate mode (new):
     If the prompt config has NO "user_prompt" key (or its "path" is absent),
     the user prompt is constructed automatically from the prompt's
     "variable_map". Each variable becomes a labeled section in a clean
     JSON-like block:

       {
         "VARIABLE_NAME": <value>,
         "OTHER_VARIABLE": <value>,
         ...
       }

     Variable names are uppercased for readability. Values are rendered
     as their string representation. None values render as empty string.

     This means new prompt stages need only a system prompt file and a
     config entry — no user prompt template file required.

The two modes are transparent to callers — render() returns a string
either way. SysPrompt is unchanged.
"""

import json as _json
from pathlib import Path
from typing import Any, Optional

from state_pipeline.prompt_management.prompt_utils import render_template


class SysPrompt:
    """Renders system prompts from config-declared text files. Unchanged."""

    def __init__(self, prompts_config: dict):
        self._specs = {
            name: spec["system_prompt"]
            for name, spec in prompts_config.items()
            if "system_prompt" in spec
        }

    def render(self, prompt_name: str, variables: dict) -> str:
        spec = self._specs[prompt_name]
        required = spec.get("variables", [])
        values = {var: variables.get(var) for var in required}
        return render_template(spec["path"], values)


class UserPrompt:
    """
    Renders user prompts from template files or auto-generates from variable_map.

    Template mode: config has "user_prompt": {"path": "...", "variables": [...]}
    Auto mode:     config has no "user_prompt" key, or path is absent/empty
    """

    def __init__(self, prompts_config: dict):
        self._prompts_config = prompts_config
        # Index template specs for prompts that declare a user_prompt path
        self._template_specs = {
            name: spec["user_prompt"]
            for name, spec in prompts_config.items()
            if "user_prompt" in spec
            and spec["user_prompt"].get("path", "").strip()
        }

    def render(self, prompt_name: str, variables: dict) -> str:
        if prompt_name in self._template_specs:
            return self._render_from_template(prompt_name, variables)
        return self._render_auto(prompt_name, variables)

    # ------------------------------------------------------------------
    # Template rendering (existing behavior)
    # ------------------------------------------------------------------

    def _render_from_template(self, prompt_name: str, variables: dict) -> str:
        spec = self._template_specs[prompt_name]
        required = spec.get("variables", [])
        values = {var: variables.get(var) for var in required}
        return render_template(spec["path"], values)

    # ------------------------------------------------------------------
    # Auto-generation from variable_map
    # ------------------------------------------------------------------

    def _render_auto(self, prompt_name: str, variables: dict) -> str:
        """
        Build a user prompt from the variable_map declared in the config.

        Produces a labeled JSON object where each key is the uppercased
        variable name and each value is the resolved state value.

        Multiline string values (like entity_registry_block or world_state)
        are rendered inline with their newlines preserved inside the JSON
        string — the LLM receives them as readable labeled sections.

        Example output for a stage with variable_map:
          {"world_state": "world_state", "scene_summary": "scene_summary"}

        Renders as:
          {
            "WORLD_STATE": "..current world state prose..",
            "SCENE_SUMMARY": "..current scene summary.."
          }
        """
        cfg = self._prompts_config.get(prompt_name, {})
        variable_map = cfg.get("variable_map", {})

        if not variable_map:
            return "{}"

        obj = {}
        for template_key in variable_map:
            value = variables.get(template_key)
            # Represent None as empty string, everything else as its string form
            if value is None:
                obj[template_key.upper()] = ""
            elif isinstance(value, (dict, list)):
                # Already structured data — embed as JSON
                obj[template_key.upper()] = value
            else:
                obj[template_key.upper()] = str(value)

        return _json.dumps(obj, indent=2, ensure_ascii=False)
