from __future__ import annotations

import argparse

from .config import load_config
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CycleGAN or LumiCycle with safe overnight resumption"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--max-hours", type=float, default=None, help="Stop cleanly after this many hours"
    )
    parser.add_argument("--resume", default="auto", help="auto, none, or a checkpoint path")
    parser.add_argument("--save-every", type=int, default=None, help="Override checkpoint interval")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--pilot", action="store_true", help="Run an isolated 2,000-step pilot")
    parser.add_argument(
        "--overfit", action="store_true", help="Overfit 32+32 images as a pipeline check"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.save_every is not None:
        config["train"]["save_every"] = args.save_every
    if args.max_steps is not None:
        config["train"]["max_steps"] = args.max_steps
    if args.pilot:
        config["train"]["max_steps"] = min(2000, int(config["train"]["max_steps"]))
        config["experiment"]["output_dir"] = f"runs/pilots/{config['experiment']['name']}"
    if args.overfit:
        config["data"]["limit"] = 32
        config["train"]["max_steps"] = min(1000, int(config["train"]["max_steps"]))
        config["experiment"]["output_dir"] = f"runs/overfit/{config['experiment']['name']}"
    Trainer(config, resume=args.resume).run(max_hours=args.max_hours)


if __name__ == "__main__":
    main()
