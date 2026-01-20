"""List all variants of a task (different config suffixes)."""

import argparse
import json
from datetime import datetime
from pathlib import Path


def list_task_variants(run_name: str, task_base: str):
    """List all variants of a task with different configs."""
    triplets_dir = Path("outputs") / run_name / "triplets"

    if not triplets_dir.exists():
        print(f"✗ Triplets directory not found: {triplets_dir}")
        return

    # Find all matching directories
    pattern = f"{task_base}__*"
    matches = sorted(triplets_dir.glob(pattern))

    if not matches:
        print(f"✗ No variants found for {task_base}")
        return

    print(f"\nFound {len(matches)} variant(s) of {task_base}:\n")
    print("=" * 80)

    for task_dir in matches:
        # Extract suffix from directory name
        task_name = task_dir.name
        suffix_parts = task_name.split("__")[2:]  # Skip docset and criterion
        config_suffix = "__".join(suffix_parts) if suffix_parts else "no suffix"

        # Load config if available
        config_file = task_dir / "triplet_config.json"
        config_info = "No config file"
        num_triplets = "?"

        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                config_info = json.dumps(config, indent=2)

        # Load triplets to count
        triplets_file = task_dir / "triplets.json"
        if triplets_file.exists():
            with open(triplets_file) as f:
                triplets = json.load(f)
                num_triplets = len(triplets)

        # Get modification time
        mod_time = datetime.fromtimestamp(task_dir.stat().st_mtime)
        mod_time_str = mod_time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{task_name}")
        print(f"  Config suffix: {config_suffix}")
        print(f"  Triplets: {num_triplets}")
        print(f"  Modified: {mod_time_str}")
        print(f"\nConfig:\n{config_info}\n")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="List all variants of a task with different config suffixes"
    )
    parser.add_argument("--run-name", required=True, help="Run name")
    parser.add_argument(
        "--task", required=True, help="Task base name (e.g., gsm8k__arithmetic)"
    )

    args = parser.parse_args()
    list_task_variants(args.run_name, args.task)


if __name__ == "__main__":
    main()
