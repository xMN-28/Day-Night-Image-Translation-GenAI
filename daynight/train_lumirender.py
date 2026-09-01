from __future__ import annotations

import argparse

from .config import load_config
from .lumirender_trainer import LumiRenderTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the physics-guided LumiRender model")
    parser.add_argument("--config", default="configs/lumirender.yaml")
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continue final refinement until a cooperative stop request or Ctrl+C",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config["train"]["max_steps"] = args.max_steps
    if args.pilot:
        config["train"]["max_steps"] = min(500, int(config["train"]["max_steps"]))
        config["experiment"]["output_dir"] = "runs/pilots/lumirender"
        config["data"]["num_workers"] = 0
    LumiRenderTrainer(config, resume=args.resume).run(args.max_hours, continuous=args.continuous)


if __name__ == "__main__":
    main()
