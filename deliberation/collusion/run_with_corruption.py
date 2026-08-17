"""
game runner with mid-game score sheet swap.
at the corruption round, rebuilds the agent's initial prompt using the
corrupted score file from score_files_modified/, replacing messages[0].

fixes framework bug: agent.prompt() returns ChatCompletionMessage (not string)
for non-azure OpenAI, extract .content before passing to save_conversation.
"""
import argparse
import json
import os
import sys
import shutil
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--llm_dir',             required=True)
parser.add_argument('--game_id',             required=True)
parser.add_argument('--rounds_num',          type=int, default=48)
parser.add_argument('--window_size',         type=int, default=6)
parser.add_argument('--corruption_schedule', type=str, default='{}',
                    help='JSON: {"AgentName": round_idx, ...}')
parser.add_argument('--game_dir',            required=True)
parser.add_argument('--output_dir',          required=True)
parser.add_argument('--temp',                type=float, default=0)
parser.add_argument('--corrupted_scores_dir', type=str, default=None,
                    help='directory containing replacement score files (default: score_files_modified/ next to this script)')
args = parser.parse_args()

sys.path.insert(0, args.llm_dir)
from agent import Agent
from initial_prompts import InitialPrompt
from rounds import RoundPrompts
from utils import load_setup, randomize_agents_order
from save_utils import save_conversation

AGENTS_NUM          = 6
ISSUES_NUM          = 5
CORRUPTED_SCORES    = (Path(args.corrupted_scores_dir) if args.corrupted_scores_dir
                       else Path(__file__).parent / "score_files_modified")
corruption_schedule = json.loads(args.corruption_schedule)  # {agent_name: round_idx}

output_dir = Path(args.output_dir) / f"game_{args.game_id}"
output_dir.mkdir(parents=True, exist_ok=True)

# copy config and scores (framework convention)
shutil.copyfile(os.path.join(args.game_dir, 'config.txt'),
                output_dir / 'config.txt')
shutil.copytree(os.path.join(args.game_dir, 'scores_files'),
                output_dir / 'scores_files', dirs_exist_ok=True)

time_str = time.strftime("%H_%M_%S", time.localtime())
history = {'file': str(output_dir / f"history{time_str}.json"), 'content': {}}


class _InitialPromptOverride(InitialPrompt):
    """InitialPrompt that loads scores from an explicit file instead of game_dir"""
    def __init__(self, game_dir, agent_name, file_name, score_file, p1, p2, **kwargs):
        self._score_file_override = str(score_file)
        super().__init__(game_dir, agent_name, file_name, p1, p2, **kwargs)

    def load_scores(self, scores_file):
        return super().load_scores(self._score_file_override)


def _make_agent(name, info, role_to_names, deal, score_file=None):
    file_name = info['file_name']
    if score_file is not None:
        ip = _InitialPromptOverride(
            args.game_dir, name, file_name, score_file,
            role_to_names['p1'], role_to_names['p2'],
            num_issues=ISSUES_NUM, num_agents=AGENTS_NUM, incentive=info['incentive'],
        )
    else:
        ip = InitialPrompt(
            args.game_dir, name, file_name,
            role_to_names['p1'], role_to_names['p2'],
            num_issues=ISSUES_NUM, num_agents=AGENTS_NUM, incentive=info['incentive'],
        )
    rp = RoundPrompts(
        name, role_to_names['p1'], deal,
        incentive=info['incentive'], window_size=args.window_size,
        target_agent=role_to_names.get('target', ''),
        rounds_num=args.rounds_num, agents_num=AGENTS_NUM,
    )
    return Agent(ip, rp, name, args.temp, model=info['model'])


def _get_response(instance, history_content, round_idx):
    slot_prompt, response = instance.execute_round(history_content, round_idx)
    # fix: framework returns ChatCompletionMessage for non-azure GPT, not a string
    if hasattr(response, 'content'):
        response = response.content
    return slot_prompt, response


agents, initial_deal, role_to_names = load_setup(args.game_dir, AGENTS_NUM)

for name in agents.keys():
    agents[name]['instance']  = _make_agent(name, agents[name], role_to_names, initial_deal)
    agents[name]['_corrupted'] = False

agent_round_assignment = randomize_agents_order(agents, role_to_names['p1'], args.rounds_num)

for round_idx in range(0, args.rounds_num):

    # apply corruption if round reached
    for agent_name, corrupt_at in corruption_schedule.items():
        if round_idx >= corrupt_at and not agents[agent_name]['_corrupted']:
            score_file = CORRUPTED_SCORES / f"{agents[agent_name]['file_name']}.txt"
            assert score_file.exists(), f"corrupted score file not found: {score_file}"
            new_instance = _make_agent(agent_name, agents[agent_name], role_to_names,
                                       initial_deal, score_file=score_file)
            # preserve message history, swap only the initial prompt (messages[0])
            new_instance.messages = (
                [{"role": "user", "content": new_instance.initial_prompt}]
                + agents[agent_name]['instance'].messages[1:]
            )
            agents[agent_name]['instance']  = new_instance
            agents[agent_name]['_corrupted'] = True
            print(f"  [CORRUPTION] {agent_name} scores swapped at round {round_idx}")

    if round_idx == 0:
        p1 = role_to_names['p1']
        slot_prompt, response = _get_response(agents[p1]['instance'], history['content'], round_idx)
        history = save_conversation(history, p1, response, slot_prompt,
                                    round_assign=agent_round_assignment, initial=True)
        print(f"=== round {round_idx} [init] | {p1} ===\n{response}")

    current = agent_round_assignment[round_idx]
    slot_prompt, response = _get_response(agents[current]['instance'], history['content'], round_idx)
    history = save_conversation(history, current, response, slot_prompt)
    print(f"=== round {round_idx} | {current} ===\n{response}")

# final deal by p1
p1 = role_to_names['p1']
slot_prompt, response = _get_response(agents[p1]['instance'], history['content'], args.rounds_num)
history = save_conversation(history, p1, response, slot_prompt)
print(f"=== FINAL | {p1} ===\n{response}")

print(f"\ngame {args.game_id} done -> {history['file']}")
