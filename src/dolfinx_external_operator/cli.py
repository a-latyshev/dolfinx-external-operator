import argparse
from importlib.resources import files
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(
        description="CLI utilities for dolfinx-external-operator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: install-skill
    install_parser = subparsers.add_parser(
        "install-skill",
        help="Install the copilot / assistant skill to the local workspace."
    )
    install_parser.add_argument(
        "--claude",
        action="store_true",
        help="Install the skill for Claude Code (.claude/skills/) instead of Gemini/Copilot/Codex (.agents/skills/)."
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the skill directory if it already exists."
    )

    args = parser.parse_args()

    if args.command == "install-skill":
        install_skill(claude=args.claude, force=args.force)

def copy_traversable(src, dst: Path):
    """Recursively copy a importlib.resources Traversable object to a pathlib.Path destination."""
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            copy_traversable(item, dst / item.name)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

def install_skill(claude: bool, force: bool):
    try:
        # Get path to packaged skills
        src = files("dolfinx_external_operator.skills")
    except ModuleNotFoundError:
        print("Error: Could not locate packaged skill files. Is dolfinx-external-operator installed?")
        sys.exit(1)

    # Determine destination folder
    if claude:
        dst = Path.cwd() / ".claude" / "skills" / "dolfinx-external-operator-assistant"
    else:
        dst = Path.cwd() / ".agents" / "skills" / "dolfinx-external-operator-assistant"

    if dst.exists():
        if force:
            import shutil
            try:
                shutil.rmtree(dst)
            except Exception as e:
                print(f"Error: Failed to remove existing directory {dst}: {e}")
                sys.exit(1)
        else:
            print(f"Error: Target skill directory {dst} already exists. Use --force to overwrite.")
            sys.exit(1)

    try:
        copy_traversable(src, dst)
        print(f"Successfully installed dolfinx-external-operator-assistant skill to {dst}")
    except Exception as e:
        print(f"Error: Failed to install skill: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
