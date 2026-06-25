import json
import logging
import os
import re
import ast
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from state_pipeline.state_store import StateStore
from state_pipeline.session_store import SessionStore
from state_pipeline.prompt_management.prompt_renderers import SysPrompt, UserPrompt
from state_pipeline.integrations.ollama_client import OllamaClient
from state_pipeline.graph_runner import GraphRunner, GraphConfig

logger = logging.getLogger(__name__)


# =============================================================================
# JSON PARSING HELPER
# =============================================================================

def robust_parse_json(text: Any) -> Any:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return text

    text = text.strip()
    text = re.sub(r"^[^\{\[]*", "", text)
    text = re.sub(r"[^\}\]]*$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(text.replace("'", '"'))
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(match.group(0))
            except Exception:
                pass

    return text


# =============================================================================
# PERSONA HELPERS
# =============================================================================

def _style_phrase(style: list) -> str:
    if not style:
        return "in their own way"
    if len(style) == 1:
        return style[0]
    return f"{', '.join(style[:-1])} and {style[-1]}"


def _list_str(items: list) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _synthesise_persona_priors(agent_data: dict) -> str:
    name     = agent_data.get("name", "The character")
    traits   = [str(t) for t in agent_data.get("traits", [])]
    style    = [str(s) for s in agent_data.get("style", [])]
    likes    = [str(l) for l in agent_data.get("likes", [])]
    dislikes = [str(d) for d in agent_data.get("dislikes", [])]
    goals    = [str(g) for g in agent_data.get("long_term_goals", [])]

    sentences = []

    if traits and style:
        joined_traits = ", ".join(traits[:-1]) + " and " + traits[-1] if len(traits) > 1 else traits[0]
        sentences.append(
            f"{name} moves through conversations {_style_phrase(style)} — "
            f"they tend to be {joined_traits}."
        )
    elif traits:
        joined = ", ".join(traits[:-1]) + " and " + traits[-1] if len(traits) > 1 else traits[0]
        sentences.append(f"{name} tends to be {joined}.")
    elif style:
        sentences.append(f"{name} engages in a {_style_phrase(style)} way.")

    if likes:
        joined = ", ".join(likes[:-1]) + " and " + likes[-1] if len(likes) > 1 else likes[0]
        sentences.append(f"They are drawn toward {joined}.")

    if dislikes:
        joined = ", ".join(dislikes[:-1]) + " and " + dislikes[-1] if len(dislikes) > 1 else dislikes[0]
        sentences.append(f"They pull back when conversations involve {joined}.")

    if goals:
        if len(goals) == 1:
            sentences.append(f"Underneath everything, they are working toward {goals[0]}.")
        else:
            sentences.append(
                f"Underneath everything, they are working toward {goals[0]}, "
                f"while also navigating {goals[1]}."
            )

    if not sentences:
        return f"{name} is a person without strongly documented tendencies."

    return " ".join(sentences)


def _derive_initial_dynamic_state(agent_data: dict) -> dict:
    traits = [str(t) for t in agent_data.get("traits", [])]
    goals  = [str(g) for g in agent_data.get("long_term_goals", [])]
    style  = [str(s) for s in agent_data.get("style", [])]
    name   = agent_data.get("name", "The character")

    guarded = [t for t in traits if any(w in t.lower()
        for w in ["guard", "caut", "reserved", "private", "wary", "skeptic", "cynic"])]
    curious = [t for t in traits if any(w in t.lower()
        for w in ["curious", "open", "warm", "eager", "interest", "thoughtful"])]

    if guarded and curious:
        current_mood = (
            "cautious but quietly curious — the familiar tension between "
            "wanting something to be interesting and not wanting to be wrong about it"
        )
    elif guarded:
        current_mood = (
            "guarded at the threshold — present but holding back, "
            "watching to see if this is worth the investment"
        )
    elif curious:
        current_mood = (
            "open but unanchored — the mild restlessness of someone "
            "who wants something to happen but doesn't yet know what"
        )
    elif traits:
        current_mood = f"at baseline — {traits[0]}, watching how this unfolds"
    else:
        current_mood = "neutral at threshold — no situational colour yet"

    if style and guarded:
        emotional_posture = (
            f"{name} enters this carefully — {style[0]} in manner, "
            f"observing before committing to anything"
        )
    elif style:
        emotional_posture = (
            f"{name} enters in their characteristic {style[0]} way — "
            f"present but not yet showing their hand"
        )
    else:
        emotional_posture = (
            f"{name} is at their baseline — no specific relational history "
            f"has shaped their stance yet"
        )

    active_desire = (
        f"{name} wants to find out whether this person is worth opening up to — "
        f"not to decide immediately, but to get enough signal to know"
    )

    return {
        "current_mood":      current_mood,
        "emotional_posture": emotional_posture,
        "active_desire":     active_desire,
        "trust_level":       "low",
        "dominance_mode":    "lead",
    }


def _load_agent_data_into_state(agent_data: dict, state: Any) -> None:
    state.set("traits",          agent_data.get("traits", []))
    state.set("likes",           agent_data.get("likes", []))
    state.set("dislikes",        agent_data.get("dislikes", []))
    state.set("skills",          agent_data.get("skills", []))
    state.set("constraints",     agent_data.get("constraints", []))
    state.set("style",           agent_data.get("style", []))
    state.set("long_term_goals", agent_data.get("long_term_goals", []))
    state.set("name",            agent_data.get("name", ""))
    state.set("persona_priors",  _synthesise_persona_priors(agent_data))

    agent_type = agent_data.get("agent_type", "")
    if agent_type:
        state.set("agent_type", agent_type)

    initial = _derive_initial_dynamic_state(agent_data)
    for key, value in initial.items():
        state.ensure(key, value)

    state.ensure("primary_emotion",     "")
    state.ensure("appraisal_intensity", "medium")
    state.ensure("expression_style",    "")
    state.ensure("internal_note",       "")


# =============================================================================
# CORE PIPELINE
# =============================================================================

class CorePipeline:
    """
    Minimal single-agent conversation pipeline.

    Usage:
        pipeline = CorePipeline()
        response = pipeline.run(
            user_input = "Hello",
            chat_id    = "session_001",
            character  = "sara",
            username   = "dennis",
        )
    """

    def __init__(
        self,
        ollama_url:         Optional[str] = None,
        model:              Optional[str] = None,
        graphs_config_path: Optional[str] = None,
        agent_type:         str = "social_conversation",
        prompts_base_dir:   Optional[str] = None,
        agents_dir:         Optional[str] = None,
        scenario_dir:       Optional[str] = None,
        users_dir:          Optional[str] = None,
        session_dir:        Optional[str] = None,
        ollama_timeout:     float = 120.0,
    ):
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.model      = model      or os.environ.get("OLLAMA_MODEL", "sara-base2:latest")

        data_root = os.environ.get("OWUI_STATE_DATA_ROOT", "./state_data")

        self.graphs_config_path = graphs_config_path or os.path.join(data_root, "graphs.json")
        self.prompts_base_dir   = prompts_base_dir   or os.path.join(data_root, "prompts")
        self.agents_dir         = agents_dir         or os.path.join(data_root, "agents")
        self.scenario_dir       = scenario_dir       or os.path.join(data_root, "scenarios")
        self.users_dir          = users_dir          or os.path.join(data_root, "users")

        # LLM defaults — used when a graph config doesn't specify its own
        self.default_max_tokens     = 1024
        self.default_temperature    = 0.7
        self.default_top_p          = 0.9
        self.default_top_k          = 40
        self.default_repeat_penalty = 1.1
        self.default_seed           = -1

        self.ollama   = OllamaClient(base_url=self.ollama_url, timeout=ollama_timeout)
        self.sessions = SessionStore(session_dir)

        # Prompt renderers are created per graph_config in _call_stage
        # (each graph has its own prompts_cfg)
        self._renderers: Dict[str, tuple] = {}  # graph_name -> (SysPrompt, UserPrompt)

        # GraphRunner — owns graph ordering and dependency resolution
        self.runner = GraphRunner(
            graphs_config_path = self.graphs_config_path,
            agent_type         = agent_type,
            call_stage         = self._call_stage,
            run_hooks          = self._run_hooks,
        )

        # Active state — set per run() call
        self.state:    Any = None
        self._chat_id: Optional[str] = None

        # Turn gate — in-memory only.
        # True  = primary loop is allowed to run (background is idle or done).
        # False = background is running, primary must wait.
        self._ready = True
        self._ready_event = threading.Event()
        self._ready_event.set()  # starts in ready state

    # =========================================================================
    # PUBLIC ENTRY POINT
    # =========================================================================

    def run(
        self,
        user_input: Optional[str],
        chat_id:    str,
        character:  Optional[str] = None,
        username:   Optional[str] = None,
        scenario:   Optional[str] = None,
    ) -> str:
        # Wait for background from previous turn to finish before proceeding.
        # In practice this is almost always instant — background finishes
        # during the user's typing window.
        self._ready_event.wait()

        self._chat_id = chat_id

        # 1. Load state from disk
        self.state = self.sessions.load(chat_id)

        # 2. Bootstrap agent if not yet loaded this session
        if character and self.state.get("_agent_loaded") != character:
            self._load_agent(character)

        # 3. Load user identity if provided
        if username and self.state.get("user_name") != username:
            self._load_user(username)

        # 4. Load scenario if provided (idempotent)
        if scenario:
            self._load_scenario(scenario)

        # 5. Store user input where prompts can find it
        self.state.set("_inbox_summary",    user_input or "")
        self.state.set("latest_user_input", user_input or "")

        # 6. Run foreground — returns response to user
        final = self.runner.run_foreground(self.state)
        self._append_history(user_input or "", final, character or "agent")
        self.sessions.save(chat_id, self.state)

        # 7. Gate closed — kick off background in a thread
        self._ready_event.clear()
        threading.Thread(
            target  = self._run_background,
            args    = (chat_id, character),
            daemon  = True,
        ).start()

        return final

    def _run_background(self, chat_id: str, character: Optional[str]) -> None:
        """
        Runs background graphs after the foreground response has been returned.
        Executes during the user's reading/typing window.
        Flips the turn gate back to ready when done.
        """
        try:
            # Reload state from disk so background works on the just-saved version
            state = self.sessions.load(chat_id)
            self.runner.run_background(state)
            self.sessions.save(chat_id, state)
            logger.info("Background graphs complete for chat '%s'", chat_id)
        except Exception as exc:
            logger.error("Background graphs failed for chat '%s': %s", chat_id, exc)
        finally:
            # Always re-open the gate, even on failure
            self._ready_event.set()

    # =========================================================================
    # LLM CALL  (called by GraphRunner via call_stage callback)
    # =========================================================================

    def _call_stage(self, stage_name: str, graph_cfg: "GraphConfig", state: Any) -> Any:
        """
        Execute one prompt stage. Called by GraphRunner for every stage in
        every graph. Returns parsed output dict (or raw string on parse failure).
        """
        # Ensure state keys for this graph's prompts exist
        self._ensure_state_keys_for(graph_cfg)

        cfg       = graph_cfg.prompts_cfg[stage_name]
        variables = self._resolve_variables(cfg.get("variable_map", {}))

        sys_renderer, user_renderer = self._get_renderers(graph_cfg)
        sys_text  = sys_renderer.render(stage_name,  variables)
        user_text = user_renderer.render(stage_name, variables)

        params = cfg.get("params", {})
        response = self.ollama.send_chat(
            model  = self.model,
            messages = [
                {"role": "system", "content": sys_text},
                {"role": "user",   "content": user_text},
            ],
            max_tokens=     params.get("max_tokens",     graph_cfg.default_max_tokens),
            temperature=    params.get("temperature",    graph_cfg.default_temperature),
            top_p=          params.get("top_p",          graph_cfg.default_top_p),
            top_k=          params.get("top_k",          graph_cfg.default_top_k),
            repeat_penalty= params.get("repeat_penalty", graph_cfg.default_repeat_penalty),
            seed=           params.get("seed",           graph_cfg.default_seed),
        )

        if isinstance(response, str) and response.strip() and response.strip()[-1] != "}":
            response = response.rstrip() + "}"

        parsed = robust_parse_json(response)
        print(f"STAGE {stage_name} | raw={response[:200]} | parsed={str(parsed)[:200]}")

        # Write end_result to state
        end_result = cfg.get("end_result")
        if end_result and isinstance(parsed, dict):
            state.set(end_result, parsed)
        elif end_result and isinstance(parsed, str):
            state.set(end_result, parsed)

        logger.debug("_call_stage %s | output=%s", stage_name, str(parsed)[:200])
        return parsed

    def _get_renderers(self, graph_cfg: "GraphConfig") -> tuple:
        """Return (SysPrompt, UserPrompt) renderers for a graph config, cached."""
        key = graph_cfg.config_path
        if key not in self._renderers:
            self._renderers[key] = (
                SysPrompt(graph_cfg.prompts_cfg),
                UserPrompt(graph_cfg.prompts_cfg),
            )
        return self._renderers[key]

    # =========================================================================
    # STATE HELPERS
    # =========================================================================

    def _resolve_variables(self, variable_map: dict) -> dict:
        resolved = {}
        for template_key, state_ref in variable_map.items():
            state_key = state_ref.replace("self.", "")
            resolved[template_key] = self.state.get(state_key)
        return resolved

    def _ensure_state_keys_for(self, graph_cfg: "GraphConfig") -> None:
        """Ensure all state keys referenced by a graph's prompts exist."""
        for prompt in graph_cfg.prompts_cfg.values():
            for ref in prompt.get("variable_map", {}).values():
                self.state.ensure(ref.replace("self.", ""))
            end_result = prompt.get("end_result")
            if end_result:
                self.state.ensure(end_result.replace("self.", ""))

    # =========================================================================
    # HOOKS  (called by GraphRunner via run_hooks callback)
    # =========================================================================

    def _run_hooks(
        self,
        phase:      str,
        stage_name: str,
        output:     Any,
        graph_cfg:  "GraphConfig",
        state:      Any,
    ) -> None:
        hooks = graph_cfg.prompts_cfg.get(stage_name, {}).get("hooks", {}).get(phase, {})
        if not hooks:
            return

        if phase == "before":
            # Add pre-prompt logic here as needed (e.g. memory retrieval)
            pass

        if phase == "after":
            # Write LLM output fields back to state
            updates = hooks.get("state_updates", {})
            if updates and isinstance(output, dict):
                for output_key, state_key in updates.items():
                    value = output.get(output_key)
                    if value is not None:
                        print(state_key, ": ", value)
                        state.set(state_key, value)
                        logger.debug("hook state_update: %s → state[%s]", output_key, state_key)

            # Working memory update — placeholder for memory implementation
            if hooks.get("update_working_memory"):
                self._update_working_memory(output, state)


    def _update_working_memory(self, output: Any, state: Any) -> None:
        """
        Placeholder for working memory update.
        Called after response_generator when update_working_memory hook is true.
        Replace with actual memory implementation.
        """
        pass

    # =========================================================================
    # HISTORY
    # =========================================================================

    def _append_history(self, user_input: str, bot_response: str, agent_name: str) -> None:
        history: list = self.state.get("_history") or []
        username = self.state.get("user_name") or "user"
        name     = self.state.get("name")      or agent_name

        history.append({"speaker": username, "role": "user",  "content": user_input})
        history.append({"speaker": name,     "role": "agent", "content": bot_response})

        self.state.set("_history", history)

    # =========================================================================
    # AGENT / USER / SCENARIO LOADING
    # =========================================================================

    def _load_agent(self, agent_id: str) -> None:
        path = os.path.join(self.agents_dir, f"{agent_id}.json")
        if not os.path.exists(path):
            logger.warning("Agent config not found: %s", path)
            self.state.set("_agent_loaded", agent_id)
            return

        data = self._load_json(path)
        _load_agent_data_into_state(data, self.state)
        self.state.set("_agent_json",   data)
        self.state.set("_agent_loaded", agent_id)
        logger.info("Loaded agent: %s", agent_id)

    def _load_user(self, username: str) -> None:
        path = os.path.join(self.users_dir, f"{username}.json")
        data = self._load_json(path)
        user = data.get("user", data)

        self.state.set("user_name",    user.get("user_name",    username))
        self.state.set("user_purpose", user.get("user_purpose", ""))
        self.state.set("user_json",    user)
        logger.info("Loaded user: %s", user.get("user_name", username))

    def _load_scenario(self, scenario_name: str) -> None:
        if self.state.get("_scenario") == scenario_name:
            return

        path = os.path.join(self.scenario_dir, f"{scenario_name}.json")
        data = self._load_json(path)

        self.state.set("medium",      data.get("medium", ""))
        self.state.set("scenario_id", data.get("id", scenario_name))
        self.state.set("_scenario",   scenario_name)

        world_state_seed = data.get("world_state_seed", "")
        if not self.state.get("world_state") and world_state_seed:
            self.state.set("world_state", world_state_seed)

        logger.info("Loaded scenario: %s", scenario_name)

    # =========================================================================
    # FILE I/O
    # =========================================================================

    def _load_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_agents(self) -> List[str]:
        return [p.stem for p in Path(self.agents_dir).glob("*.json")]

    def list_scenarios(self) -> List[str]:
        return [p.stem for p in Path(self.scenario_dir).glob("*.json")]

    def list_users(self) -> List[str]:
        return [p.stem for p in Path(self.users_dir).glob("*.json")]