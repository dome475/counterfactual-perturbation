"""
split each mid-game-swap log into two windowed sub-games:
window_a before the swap (rounds 1-12), window_b after (rounds 13-24).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from extract_features import (
    _load_scores, compute_ideal, extract_game,
    AGENT_KEYS, ISSUE_SIZES, NAME_MAP, SCORES_DIR,
)

LOGS_DIR = Path(__file__).parent.parent / "logs" / "ceo_update"
OUT_DIR  = Path(__file__).parent / "features"
SPLIT_T  = 13


def split_game(game, split_t=SPLIT_T):
    base = {k: v for k, v in game.items() if k != "rounds"}
    early = [r for r in game["rounds"] if r["t"] < split_t]
    late  = [r for r in game["rounds"] if r["t"] >= split_t]
    wa = {**base, "rounds": early, "n_rounds_parsed": len(early), "window": "A"}
    wb = {**base, "rounds": late,  "n_rounds_parsed": len(late),  "window": "B"}
    return wa, wb


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores, _ = _load_scores(SCORES_DIR, AGENT_KEYS, ISSUE_SIZES)

    agent_meta = {}
    for a in AGENT_KEYS:
        opts, ideal_sc, worst_sc = compute_ideal(scores, a, ISSUE_SIZES)
        agent_meta[a] = {"ideal_opts": opts, "ideal_score": ideal_sc, "worst_score": worst_sc}
        print(f"  {a:<15} ideal={ideal_sc}  worst={worst_sc}")

    files = sorted(LOGS_DIR.glob("ceo_*.json"))
    if not files:
        print(f"no files in {LOGS_DIR}")
        return

    print(f"\n{len(files)} game files")
    for path in files:
        game  = extract_game(path, "ceo_update", scores, agent_meta,
                             AGENT_KEYS, ISSUE_SIZES, NAME_MAP)
        wa, wb = split_game(game)
        out = {
            "game_id":         game["game_id"],
            "agents":          game["agents"],
            "ideal_score":     game["ideal_score"],
            "worst_score":     game["worst_score"],
            "ideal_opts":      game["ideal_opts"],
            "n_rounds_total":  game["n_rounds_parsed"],
            "parse_failures":  game["parse_failures"],
            "split_t":         SPLIT_T,
            "window_a":        wa,
            "window_b":        wb,
        }
        out_path = OUT_DIR / f"{game['game_id']}.json"
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"  {game['game_id']:<20} total={game['n_rounds_parsed']}  "
              f"A={len(wa['rounds'])}  B={len(wb['rounds'])}")

    print(f"\nfeatures -> {OUT_DIR}")
    print("next: python analyze.py")


if __name__ == "__main__":
    main()
