from __future__ import annotations

import argparse

from .common import load_config
from .hdfs import run_data_process, run_predict, run_train, run_vocab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the public-repo LogBERT HDFS flow inside this repository."
    )
    parser.add_argument(
        "--config",
        default="configs/related_work/logbert_hdfs_public.json",
        help="Path to the dedicated LogBERT public-repo config.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("data_process")
    subparsers.add_parser("train")
    subparsers.add_parser("all")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("-m", "--mean", type=float, default=0.0)
    predict_parser.add_argument("-s", "--std", type=float, default=1.0)

    vocab_parser = subparsers.add_parser("vocab")
    vocab_parser.add_argument("-s", "--vocab_size", type=int, default=None)
    vocab_parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    vocab_parser.add_argument("-m", "--min_freq", type=int, default=1)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "data_process":
        run_data_process(config)
    elif args.mode == "vocab":
        run_vocab(config, args.vocab_size, args.encoding, args.min_freq)
    elif args.mode == "train":
        run_train(config)
    elif args.mode == "predict":
        run_predict(config, args.mean, args.std)
    elif args.mode == "all":
        run_data_process(config)
        run_vocab(config, None, "utf-8", 1)
        run_train(config)
        run_predict(config, 0.0, 1.0)


if __name__ == "__main__":
    main()
