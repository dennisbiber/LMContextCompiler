import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# GRAPH CONFIG LOADER
# =============================================================================

class GraphConfig:
    """
    Holds the pipeline config for a single graph (e.g. main_loop.json).
    Identical structure to what CorePipeline already reads for main_loop —
    prompts, order, llm_defaults, response_key.
    """

    def __init__(self, config: dict, config_path: str):
        self.config       = config
        self.config_path  = config_path
        self.prompts_cfg  = config["prompts"]
        self.response_key = config.get("response_key")
        self.stage_order  = self._load_stage_order()

        llm = config.get("llm_defaults", {})
        self.default_max_tokens     = llm.get("max_tokens",     1024)
        self.default_temperature    = llm.get("temperature",    0.7)
        self.default_top_p          = llm.get("top_p",          0.9)
        self.default_top_k          = llm.get("top_k",          40)
        self.default_repeat_penalty = llm.get("repeat_penalty", 1.1)
        self.default_seed           = llm.get("seed",           -1)

    def _load_stage_order(self) -> List[str]:
        order_map = self.config.get("order", {})
        if not order_map:
            return list(self.prompts_cfg.keys())
        return [order_map[k] for k in sorted(order_map.keys(), key=lambda x: int(x))]

    @classmethod
    def from_file(cls, path: str) -> "GraphConfig":
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(config, path)


# =============================================================================
# GRAPH RUNNER
# =============================================================================

class GraphRunner:
    """
    Executes foreground and background graphs for a single agent type.

    The runner is stateless between turns — all state lives in the state
    object passed to run_turn(). The runner just orchestrates which graphs
    run and in what order.

    Parameters
    ----------
    graphs_config_path : str
        Path to graphs.json
    agent_type : str
        Which agent_type block to load from graphs.json
    call_stage : Callable[[str, GraphConfig, Any], Any]
        Provided by CorePipeline. Signature:
            call_stage(stage_name, graph_config, state) -> parsed_output
    run_hooks : Callable[[str, str, Any, GraphConfig, Any], None]
        Provided by CorePipeline. Signature:
            run_hooks(phase, stage_name, output, graph_config, state)
    """

    def __init__(
        self,
        graphs_config_path: str,
        agent_type:         str,
        call_stage:         Callable,
        run_hooks:          Callable,
    ):
        self.agent_type = agent_type
        self.call_stage = call_stage
        self.run_hooks  = run_hooks

        graphs_config = self._load_json(graphs_config_path)
        agent_cfg     = graphs_config["agent_types"].get(agent_type)
        if agent_cfg is None:
            raise ValueError(
                f"GraphRunner: agent_type '{agent_type}' not found in {graphs_config_path}"
            )

        self._raw_graphs = agent_cfg["graphs"]

        # Resolve foreground graph name
        # Support both old tier-based format and new simple format
        tiers = agent_cfg.get("tiers", {})
        if tiers:
            self.foreground_name = tiers.get("0", [None])[0]
        else:
            self.foreground_name = agent_cfg.get("foreground")

        if not self.foreground_name:
            raise ValueError(
                f"GraphRunner: no foreground graph defined for agent_type '{agent_type}'"
            )

        # Resolve background graph names in dependency order
        if tiers:
            background_names = []
            for tier_key in sorted(tiers.keys(), key=int):
                if int(tier_key) == 0:
                    continue
                background_names.extend(tiers[tier_key])
        else:
            background_names = agent_cfg.get("background", [])

        self.background_names = self._sort_by_dependencies(background_names)

        # Load graph configs (lazy — only load what's needed)
        self._graph_configs: Dict[str, GraphConfig] = {}

    # =========================================================================
    # PUBLIC: RUN A TURN
    # =========================================================================

    def run_foreground(self, state: Any) -> str:
        """
        Run the foreground graph and return the user-facing response.
        Called synchronously — the user waits for this.
        """
        response = self._run_graph(self.foreground_name, state)
        logger.info("GraphRunner: foreground '%s' complete", self.foreground_name)
        return response

    def run_background(self, state: Any) -> None:
        """
        Run all background graphs in dependency order.
        Called in a thread after the foreground response is returned.
        Populates state variables consumed by the foreground on the next turn.
        """
        for graph_name in self.background_names:
            try:
                self._run_graph(graph_name, state)
                logger.info("GraphRunner: background '%s' complete", graph_name)
            except Exception as exc:
                logger.error(
                    "GraphRunner: background graph '%s' failed: %s — continuing",
                    graph_name, exc
                )

    # =========================================================================
    # GRAPH EXECUTION
    # =========================================================================

    def _run_graph(self, graph_name: str, state: Any) -> str:
        """Run all stages of a single graph in order. Returns final response string."""
        graph_cfg   = self._get_graph_config(graph_name)
        last_output = None

        for stage_name in graph_cfg.stage_order:
            if stage_name not in graph_cfg.prompts_cfg:
                logger.warning(
                    "GraphRunner: stage '%s' in order for graph '%s' but not in prompts — skipping",
                    stage_name, graph_name
                )
                continue

            self.run_hooks("before", stage_name, None, graph_cfg, state)
            output = self.call_stage(stage_name, graph_cfg, state)
            print(output)
            self.run_hooks("after",  stage_name, output, graph_cfg, state)
            last_output = output

        return self._extract_response(last_output, graph_cfg)

    def _extract_response(self, last_output: Any, graph_cfg: GraphConfig) -> str:
        if last_output is None:
            return ""
        if graph_cfg.response_key and isinstance(last_output, dict):
            reply = last_output.get(graph_cfg.response_key)
            if reply is not None:
                return str(reply)
        if isinstance(last_output, str):
            return last_output
        return ""

    # =========================================================================
    # GRAPH CONFIG LOADING
    # =========================================================================

    def _get_graph_config(self, graph_name: str) -> GraphConfig:
        if graph_name not in self._graph_configs:
            raw  = self._raw_graphs.get(graph_name)
            if raw is None:
                raise ValueError(f"GraphRunner: graph '{graph_name}' not found in graphs config")
            path = os.path.normpath(raw["config"])
            self._graph_configs[graph_name] = GraphConfig.from_file(path)
        return self._graph_configs[graph_name]


    # =========================================================================
    # DEPENDENCY SORT
    # =========================================================================

    def _sort_by_dependencies(self, names: List[str]) -> List[str]:
        """
        Topological sort of background graph names based on depends_on.
        Graphs with no dependencies come first. Cycles are broken arbitrarily.
        """
        if not names:
            return []

        deps: Dict[str, List[str]] = {}
        for name in names:
            raw  = self._raw_graphs.get(name, {})
            deps[name] = [d for d in raw.get("depends_on", []) if d in names]

        ordered = []
        visited = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in deps.get(name, []):
                visit(dep)
            ordered.append(name)

        for name in names:
            visit(name)

        return ordered

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _load_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)