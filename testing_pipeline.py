import argparse
import json
import sys

from state_pipeline.state_manager import CorePipeline

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Interactive test harness with observability.")
parser.add_argument(
    "--config",
    required=True,
    metavar="PATH",
    help="Relative path to a JSON test config file (e.g. test_configs/my_config.json)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

try:
    with open(args.config, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Config file not found: '{args.config}'")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Failed to parse config file '{args.config}': {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENTS             = config["agents"]
SCENARIO           = config["scenario"]
USERNAME           = config["username"]
CHAT_ID            = config["chat_id"]
FIRST_INPUT        = config.get("first_input", None)
USER_FACING_AGENT  = config.get("user_facing_agent", AGENTS[-1])

LOG_DIR       = "./sessions/logs"
STATE_LOG_DIR = "./sessions/state_logs"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

sm = CorePipeline(
    prompts_base_dir="./state_data/prompts",
    agents_dir="./state_data/agents",
    scenario_dir="./state_data/scenarios",
    users_dir="./state_data/users",
    session_dir="./sessions"
)


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------

first_turn = True

print(f"\nConfig:    {args.config}")
print(f"Agents:    {AGENTS}  |  Scenario: {SCENARIO}")
print(f"Type 'quit' to exit, 'reset' to clear session, 'inspect <n>' to view turn N.\n")

while True:
    # On the first turn, use first_input from config if provided
    if first_turn and FIRST_INPUT is not None:
        user_text = FIRST_INPUT
        print(f"You (auto): {user_text}")
    else:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

    if not user_text:
        continue

    if user_text.lower() == "quit":
        break

    if user_text.lower() == "reset":
        sm.sessions.delete(CHAT_ID)
        first_turn = True
        print("  [Session cleared]\n")
        continue


    response = sm.run(
        user_input         = user_text,
        chat_id            = CHAT_ID,
        username           = USERNAME,
        scenario           = SCENARIO,
        character          = USER_FACING_AGENT
    )

    first_turn = False
    print(f"\n{USER_FACING_AGENT.capitalize()}: {response}\n")
