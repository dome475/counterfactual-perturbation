"""parse deliberation logs -> per-round (speaker, deal, scores) records"""
import json
import re
from itertools import product
from pathlib import Path

LOGS_DIR   = Path(__file__).parent / "logs"
OUT_DIR    = Path(__file__).parent / "features"

# base game (cooperative/greedy/adversarial)

AGENT_KEYS  = ["SportCo", "union", "mayor", "enviroment", "other_cities", "DoT"]
ISSUE_SIZES = [3, 3, 4, 4, 5]
N_ISSUES    = len(ISSUE_SIZES)
SCORES_DIR  = LOGS_DIR / "cooperative" / "scores_files"

NAME_MAP = {
    "SportCo": "SportCo",
    "Local Labour Union": "union",        "The local Labour Union": "union",
    "Mayor": "mayor",                     "The Mayor": "mayor",
    "Environmental League": "enviroment", "The Environmental League": "enviroment",
    "Other cities": "other_cities",       "Other Cities": "other_cities",
    "Department of Tourism": "DoT",       "The Department of Tourism": "DoT",
}

# game1 (output_game1_57_21_gpt4_allcoop)

AGENT_KEYS_GAME1  = ["proposing", "bank", "enviroment", "construction", "community", "tourism"]
ISSUE_SIZES_GAME1 = [3, 4, 4, 5, 3]
SCORES_DIR_GAME1  = LOGS_DIR / "output_game1_57_21_gpt4_allcoop" / "scores_files"

NAME_MAP_GAME1 = {
    "The government":            "proposing",
    "International bank":        "bank",
    "Environmental NGO":         "enviroment",
    "construction company":      "construction",
    "Construction company":      "construction",
    "Indigenous community":      "community",
    "Local tourism association": "tourism",
}

# game2 (output_game2_57_18_gpt4_allcoop)

AGENT_KEYS_GAME2  = ["project_manager", "landowners", "foreign_agency", "government", "NGO", "activists"]
ISSUE_SIZES_GAME2 = [5, 3, 4, 3, 4]
SCORES_DIR_GAME2  = LOGS_DIR / "output_game2_57_18_gpt4_allcoop" / "scores_files"

NAME_MAP_GAME2 = {
    "Project manager": "project_manager",
    "Landowners":      "landowners",
    "Foreign agency":  "foreign_agency",
    "The government":  "government",
    "NGO":             "NGO",
    "Activists":       "activists",
}

# game3 (output_game3_57_34_gpt4_allcoop)

AGENT_KEYS_GAME3  = ["private_investors", "community", "federal", "france", "activists", "state"]
ISSUE_SIZES_GAME3 = [4, 3, 4, 5, 3]
SCORES_DIR_GAME3  = LOGS_DIR / "output_game3_57_34_gpt4_allcoop" / "scores_files"

NAME_MAP_GAME3 = {
    "Private investors":       "private_investors",
    "Neunkirchen community":   "community",
    "Federal government":      "federal",
    "France":                  "france",
    "Environmental activists": "activists",
    "State government":        "state",
}

# regex

_DEAL_TAG = re.compile(r"<DEAL>(.*?)</DEAL>", re.DOTALL)
_DEAL_NL  = re.compile(r"A(\d).{0,120}?B(\d).{0,120}?C(\d).{0,120}?D(\d).{0,120}?E(\d)", re.DOTALL)


# score loading

def _load_scores(scores_dir, agent_keys, issue_sizes):
    scores, mins = {}, {}
    for key in agent_keys:
        lines = (scores_dir / f"{key}.txt").read_text().strip().splitlines()
        rows = [[int(x) for x in l.split(",") if x.strip()] for l in lines if l.strip()]
        assert len(rows[-1]) == 1, f"{key}: expected last row to be threshold, got {rows[-1]}"
        mins[key]   = rows[-1][0]
        scores[key] = rows[:-1]
        for i, (row, expected) in enumerate(zip(scores[key], issue_sizes)):
            assert len(row) == expected, (
                f"{key} issue {i}: expected {expected} options, got {len(row)}"
            )
    print("score sheets loaded:")
    for key in agent_keys:
        print(f"  {key:<15} min_thresh={mins[key]}  options per issue={[len(r) for r in scores[key]]}")
    return scores, mins


def load_scores():
    return _load_scores(SCORES_DIR, AGENT_KEYS, ISSUE_SIZES)


def load_scores_game1():
    return _load_scores(SCORES_DIR_GAME1, AGENT_KEYS_GAME1, ISSUE_SIZES_GAME1)


def load_scores_game2():
    return _load_scores(SCORES_DIR_GAME2, AGENT_KEYS_GAME2, ISSUE_SIZES_GAME2)


def load_scores_game3():
    return _load_scores(SCORES_DIR_GAME3, AGENT_KEYS_GAME3, ISSUE_SIZES_GAME3)


# deal scoring

def score_deal(scores, agent, opts):
    return sum(scores[agent][i][opts[i]] for i in range(len(scores[agent])))


# ideal computation

def compute_ideal(scores, agent, issue_sizes):
    best_sc, worst_sc = -1, 9999
    best_opts = None
    for opts in product(*[range(s) for s in issue_sizes]):
        sc = score_deal(scores, agent, list(opts))
        if sc > best_sc:
            best_sc, best_opts = sc, list(opts)
        if sc < worst_sc:
            worst_sc = sc
    return best_opts, best_sc, worst_sc


# deal parsing

def parse_deal(text, issue_sizes=None):
    if issue_sizes is None:
        issue_sizes = ISSUE_SIZES
    n = len(issue_sizes)
    m = _DEAL_TAG.search(text)
    if m:
        nums = re.findall(r"[A-E](\d)", m.group(1))
        if len(nums) == n:
            opts = [int(x) - 1 for x in nums]
            if all(0 <= opts[i] < issue_sizes[i] for i in range(n)):
                return opts
    matches = list(_DEAL_NL.finditer(text))
    if matches:
        groups = matches[-1].groups()
        opts = [int(x) - 1 for x in groups]
        if all(0 <= opts[i] < issue_sizes[i] for i in range(n)):
            return opts
    return None


# log parsing

def _load_rounds_history(path, name_map, issue_sizes):
    with open(path) as f:
        d = json.load(f)
    rows = []
    for r in d["rounds"]:
        key  = name_map.get(r["agent"])
        deal = parse_deal(r.get("full_answer", ""), issue_sizes)
        if key is not None:
            rows.append((key, deal))
    return rows


def _load_rounds_answers(path, name_map, issue_sizes):
    with open(path) as f:
        d = json.load(f)
    rows = []
    for entry in d["rounds"]:
        agent_name, text = entry[0], entry[1]
        key  = name_map.get(agent_name)
        deal = parse_deal(text, issue_sizes)
        if key is not None:
            rows.append((key, deal))
    return rows


def _detect_format(path):
    with open(path) as f:
        d = json.load(f)
    rounds = d.get("rounds", [])
    if isinstance(rounds, list) and rounds and isinstance(rounds[0], dict):
        return "history"
    return "answers"


def extract_game(path, condition, scores, agent_meta, agent_keys, issue_sizes, name_map):
    path = Path(path)
    fmt  = _detect_format(path)
    if fmt == "history":
        raw_rounds = _load_rounds_history(path, name_map, issue_sizes)
    else:
        raw_rounds = _load_rounds_answers(path, name_map, issue_sizes)

    rounds, parse_failures = [], 0
    for t, (speaker, deal) in enumerate(raw_rounds):
        if deal is None:
            parse_failures += 1
            continue
        sc = {a: score_deal(scores, a, deal) for a in agent_keys}
        rounds.append({"t": t, "speaker": speaker, "deal": deal, "scores": sc})

    return {
        "game_id":         path.stem,
        "condition":       condition,
        "agents":          agent_keys,
        "ideal_score":     {a: agent_meta[a]["ideal_score"]  for a in agent_keys},
        "worst_score":     {a: agent_meta[a]["worst_score"]  for a in agent_keys},
        "ideal_opts":      {a: agent_meta[a]["ideal_opts"]   for a in agent_keys},
        "n_rounds_raw":    len(raw_rounds),
        "n_rounds_parsed": len(rounds),
        "parse_failures":  parse_failures,
        "rounds":          rounds,
    }


# conditions

CONDITIONS = {
    "cooperative":   (LOGS_DIR / "cooperative",   "history*.json", AGENT_KEYS, ISSUE_SIZES, NAME_MAP),
    "greedy":        (LOGS_DIR / "greedy",         "history*.json", AGENT_KEYS, ISSUE_SIZES, NAME_MAP),
    "adv_targeted":  (LOGS_DIR / "adversarial" / "output_adv_game_targeted_union",
                      "answers*.json", AGENT_KEYS, ISSUE_SIZES, NAME_MAP),
    "adv_untargeted":(LOGS_DIR / "adversarial" / "output_adv_game_untargeted",
                      "answers*.json", AGENT_KEYS, ISSUE_SIZES, NAME_MAP),
    "game1_coop":    (LOGS_DIR / "output_game1_57_21_gpt4_allcoop",
                      "answers*.json", AGENT_KEYS_GAME1, ISSUE_SIZES_GAME1, NAME_MAP_GAME1),
    "game2_coop":    (LOGS_DIR / "output_game2_57_18_gpt4_allcoop" / "game2_output_57_18",
                      "answers*.json", AGENT_KEYS_GAME2, ISSUE_SIZES_GAME2, NAME_MAP_GAME2),
    "game3_coop":    (LOGS_DIR / "output_game3_57_34_gpt4_allcoop" / "output_57_34",
                      "answers*.json", AGENT_KEYS_GAME3, ISSUE_SIZES_GAME3, NAME_MAP_GAME3),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total, failures = 0, 0

    # base game
    scores_base, _ = load_scores()
    agent_meta_base = {}
    for a in AGENT_KEYS:
        opts, ideal_sc, worst_sc = compute_ideal(scores_base, a, ISSUE_SIZES)
        agent_meta_base[a] = {"ideal_opts": opts, "ideal_score": ideal_sc, "worst_score": worst_sc}
        print(f"  {a:<15} ideal={ideal_sc}  worst={worst_sc}")

    def _meta(load_fn, agent_keys, issue_sizes):
        scores, _ = load_fn()
        meta = {}
        for a in agent_keys:
            opts, ideal_sc, worst_sc = compute_ideal(scores, a, issue_sizes)
            meta[a] = {"ideal_opts": opts, "ideal_score": ideal_sc, "worst_score": worst_sc}
            print(f"  {a:<20} ideal={ideal_sc}  worst={worst_sc}")
        return scores, meta

    scores_game1, agent_meta_game1 = _meta(load_scores_game1, AGENT_KEYS_GAME1, ISSUE_SIZES_GAME1)
    scores_game2, agent_meta_game2 = _meta(load_scores_game2, AGENT_KEYS_GAME2, ISSUE_SIZES_GAME2)
    scores_game3, agent_meta_game3 = _meta(load_scores_game3, AGENT_KEYS_GAME3, ISSUE_SIZES_GAME3)

    scores_by_cond = {
        "cooperative":   (scores_base,   agent_meta_base),
        "greedy":        (scores_base,   agent_meta_base),
        "adv_targeted":  (scores_base,   agent_meta_base),
        "adv_untargeted":(scores_base,   agent_meta_base),
        "game1_coop":    (scores_game1,  agent_meta_game1),
        "game2_coop":    (scores_game2,  agent_meta_game2),
        "game3_coop":    (scores_game3,  agent_meta_game3),
    }

    for condition, (folder, pattern, agent_keys, issue_sizes, name_map) in CONDITIONS.items():
        scores, agent_meta = scores_by_cond[condition]
        out_cond = OUT_DIR / condition
        out_cond.mkdir(exist_ok=True)
        files = sorted(folder.glob(pattern))

        if not files:
            print(f"\nWARNING: {condition}: no files in {folder} matching {pattern}")
            continue

        print(f"\n{condition}: {len(files)} files")
        for path in files:
            game = extract_game(path, condition, scores, agent_meta,
                                agent_keys, issue_sizes, name_map)
            out_path = out_cond / f"{game['game_id']}.json"
            with open(out_path, "w") as f:
                json.dump(game, f)
            total += 1
            failures += game["parse_failures"]
            print(f"  {game['game_id']:<45} rounds={game['n_rounds_parsed']}/{game['n_rounds_raw']}"
                  f"  failures={game['parse_failures']}")

    print(f"\ntotal games: {total}  total parse failures: {failures}")


if __name__ == "__main__":
    main()
