"""Command line interface for PauseNet."""

from __future__ import annotations

import argparse

from .evaluate import evaluate_checkpoint
from .prepare_bigwig import add_prepare_bigwig_args, run_prepare_bigwig_from_args
from .train import load_config, train_from_config
from .visualize import add_visualize_args, visualize_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(prog="pausenet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train PauseNet.")
    train_parser.add_argument("--config", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint.")
    eval_parser.add_argument("--data-dir", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.add_argument("--split", default="test")
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--batch-size", type=int, default=256)
    eval_parser.add_argument("--num-workers", type=int, default=4)

    prepare_parser = subparsers.add_parser(
        "prepare-bigwig",
        help="Create PauseNet split directories from strand-specific bigWig files.",
    )
    add_prepare_bigwig_args(prepare_parser)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Create count and profile-similarity figures from evaluation outputs.",
    )
    add_visualize_args(visualize_parser)

    args = parser.parse_args()
    if args.command == "train":
        best_path = train_from_config(load_config(args.config))
        print(f"Best checkpoint: {best_path}")
    elif args.command == "evaluate":
        evaluate_checkpoint(
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.command == "prepare-bigwig":
        run_prepare_bigwig_from_args(args)
    elif args.command == "visualize":
        count_path, profile_path = visualize_evaluation(
            evaluation_dir=args.evaluation_dir,
            split=args.split,
            output_dir=args.output_dir,
            image_format=args.format,
        )
        print(f"Count scatter: {count_path}")
        print(f"Profile similarity: {profile_path}")


if __name__ == "__main__":
    main()
