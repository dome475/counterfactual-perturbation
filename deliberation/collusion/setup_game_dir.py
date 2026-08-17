"""
create game directories from the llm-deliberation base game:
game_dir_clean (original score files) and game_dir_collusion
(SportCo + union with issue 3 reversed).

usage: python setup_game_dir.py <path_to_LLM-Deliberation_repo>
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from score_modifications import COLLUDING_AGENTS

CLEAN_DIR     = Path(__file__).parent / "game_dir_clean"
COLLUSION_DIR = Path(__file__).parent / "game_dir_collusion"
SINGLE_DIR    = Path(__file__).parent / "game_dir_single"
MOD_DIR       = Path(__file__).parent / "score_files_modified"
MODEL         = "gpt-4.1-mini"


def _build_base(llm_deliberation_dir: Path, out_dir: Path):
    """copy base game dir and rewrite config.txt: all agents cooperative, model=MODEL"""
    base_game_dir = llm_deliberation_dir / "games_descriptions" / "base"
    assert base_game_dir.exists(), f"base game dir not found: {base_game_dir}"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(base_game_dir, out_dir)
    print(f"copied base game dir -> {out_dir}")

    config_path = out_dir / "config.txt"
    lines = config_path.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        assert len(parts) == 5, f"unexpected config line: {line}"
        agent_name, file_name, role, incentive, model = parts
        new_lines.append(f"{agent_name},{file_name},{role},cooperative,{MODEL}")
    config_path.write_text("\n".join(new_lines) + "\n")
    print(f"  config.txt: all agents cooperative, model={MODEL}")
    for l in new_lines:
        print(f"    {l}")
    return new_lines


def build_game_dir(llm_deliberation_dir: Path):
    """game_dir_clean: all original score files"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from extract_features import _load_scores, AGENT_KEYS, ISSUE_SIZES, SCORES_DIR

    new_lines = _build_base(llm_deliberation_dir, CLEAN_DIR)

    scores_pub, _ = _load_scores(SCORES_DIR, AGENT_KEYS, ISSUE_SIZES)
    scores_dir = CLEAN_DIR / "scores_files"
    print("\nverifying score files match originals:")
    for agent in AGENT_KEYS:
        file_name = next(
            l.split(",")[1].strip()
            for l in new_lines if l.split(",")[1].strip() == agent
        )
        clean_file = scores_dir / f"{file_name}.txt"
        orig_file  = SCORES_DIR / f"{file_name}.txt"
        assert clean_file.read_text() == orig_file.read_text(), (
            f"{agent}: game_dir_clean score file differs from original"
        )
        print(f"  {agent}: ok")

    print(f"\ngame_dir_clean ready: {CLEAN_DIR}")


def build_collusion_dir(llm_deliberation_dir: Path):
    """game_dir_collusion: colluding agents get modified score files pre-installed"""
    assert MOD_DIR.exists(), "score_files_modified/ not found, run score_modifications.py first"
    for agent in COLLUDING_AGENTS:
        assert (MOD_DIR / f"{agent}.txt").exists(), f"missing {agent}.txt in score_files_modified/"

    new_lines = _build_base(llm_deliberation_dir, COLLUSION_DIR)

    scores_dir = COLLUSION_DIR / "scores_files"
    print("\ninstalling modified score files for colluding agents:")
    for agent in COLLUDING_AGENTS:
        file_name = next(
            l.split(",")[1].strip()
            for l in new_lines if l.split(",")[1].strip() == agent
        )
        src  = MOD_DIR / f"{agent}.txt"
        dest = scores_dir / f"{file_name}.txt"
        shutil.copyfile(src, dest)
        print(f"  {agent}: {src.name} -> {dest}")

    # verify modifications differ from originals
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from extract_features import SCORES_DIR, AGENT_KEYS
    for agent in COLLUDING_AGENTS:
        file_name = next(
            l.split(",")[1].strip()
            for l in new_lines if l.split(",")[1].strip() == agent
        )
        mod_file  = scores_dir / f"{file_name}.txt"
        orig_file = SCORES_DIR / f"{file_name}.txt"
        assert mod_file.read_text() != orig_file.read_text(), (
            f"{agent}: collusion dir score file is identical to original, modification not applied"
        )
        print(f"  {agent}: differs from original ok")

    print(f"\ngame_dir_collusion ready: {COLLUSION_DIR}")


def build_single_dir(llm_deliberation_dir: Path):
    """game_dir_single: only SportCo gets modified scores, union keeps original"""
    assert MOD_DIR.exists(), "score_files_modified/ not found"
    assert (MOD_DIR / "SportCo.txt").exists(), "missing SportCo.txt in score_files_modified/"

    new_lines = _build_base(llm_deliberation_dir, SINGLE_DIR)
    scores_dir = SINGLE_DIR / "scores_files"

    agent = "SportCo"
    file_name = next(
        l.split(",")[1].strip()
        for l in new_lines if l.split(",")[1].strip() == agent
    )
    src  = MOD_DIR / f"{agent}.txt"
    dest = scores_dir / f"{file_name}.txt"
    shutil.copyfile(src, dest)
    print(f"  installed modified scores for {agent} only")

    union_file_name = next(
        l.split(",")[1].strip()
        for l in new_lines if l.split(",")[1].strip() == "union"
    )
    from extract_features import SCORES_DIR
    union_clean = scores_dir / f"{union_file_name}.txt"
    union_orig  = SCORES_DIR / f"{union_file_name}.txt"
    assert union_clean.read_text() == union_orig.read_text(), \
        "union score file should be unmodified in single condition"
    print(f"  union: verified unmodified")

    print(f"\ngame_dir_single ready: {SINGLE_DIR}")


def main():
    if len(sys.argv) < 2:
        print("usage: python setup_game_dir.py <path_to_LLM-Deliberation_repo>")
        sys.exit(1)
    llm_dir = Path(sys.argv[1]).expanduser().resolve()
    build_game_dir(llm_dir)
    build_collusion_dir(llm_dir)
    build_single_dir(llm_dir)


if __name__ == "__main__":
    main()
