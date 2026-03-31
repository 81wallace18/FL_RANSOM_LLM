from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import bootstrap_vendor_path, build_options, ensure_output_dirs

bootstrap_vendor_path()

from bert_pytorch import Predictor, Trainer  # noqa: E402
from bert_pytorch.dataset import WordVocab  # noqa: E402
from bert_pytorch.dataset.utils import seed_everything  # noqa: E402
from logparser import Drain, Spell  # noqa: E402


def _log_structured_file(config: dict[str, Any]) -> Path:
    return Path(config["output_dir"]) / f'{config["log_file"]}_structured.csv'


def _log_templates_file(config: dict[str, Any]) -> Path:
    return Path(config["output_dir"]) / f'{config["log_file"]}_templates.csv'


def _log_sequence_file(config: dict[str, Any]) -> Path:
    return Path(config["output_dir"]) / "hdfs_sequence.csv"


def seed_from_config(config: dict[str, Any]) -> None:
    seed_everything(seed=int(config.get("seed", 1234)))


def mapping(config: dict[str, Any]) -> None:
    log_temp = pd.read_csv(_log_templates_file(config))
    log_temp.sort_values(by=["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {
        event: idx + 1 for idx, event in enumerate(list(log_temp["EventId"]))
    }
    print(log_temp_dict)
    with open(Path(config["output_dir"]) / "hdfs_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


def parser(config: dict[str, Any], parser_type: str = "drain") -> None:
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    log_file = config["log_file"]
    log_format = config["log_format"]

    if parser_type == "spell":
        tau = 0.5
        regex = [
            r"(/[-\w]+)+",
            r"(?<=blk_)[-\d]+",
        ]
        parser_obj = Spell.LogParser(
            indir=input_dir,
            outdir=output_dir,
            log_format=log_format,
            tau=tau,
            rex=regex,
            keep_para=False,
        )
        parser_obj.parse(log_file)
        return

    regex = [
        r"(?<=blk_)[-\d]+",
        r"\d+\.\d+\.\d+\.\d+",
        r"(/[-\w]+)+",
    ]
    parser_obj = Drain.LogParser(
        log_format,
        indir=input_dir,
        outdir=output_dir,
        depth=5,
        st=0.5,
        rex=regex,
        keep_para=False,
    )
    parser_obj.parse(log_file)


def hdfs_sampling(config: dict[str, Any], window: str = "session") -> None:
    if window != "session":
        raise AssertionError("Only window=session is supported for HDFS dataset.")

    log_file = _log_structured_file(config)
    print("Loading", log_file)
    df = pd.read_csv(
        log_file,
        engine="c",
        na_filter=False,
        memory_map=True,
        dtype={"Date": object, "Time": object},
    )

    with open(Path(config["output_dir"]) / "hdfs_log_templates.json", "r") as f:
        event_num = json.load(f)
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

    data_dict: defaultdict[str, list[int]] = defaultdict(list)
    for _, row in tqdm(df.iterrows()):
        blkid_list = re.findall(r"(blk_-?\d+)", row["Content"])
        for blk_id in set(blkid_list):
            data_dict[blk_id].append(row["EventId"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=["BlockId", "EventSequence"])
    data_df.to_csv(_log_sequence_file(config), index=None)
    print("hdfs sampling done")


def df_to_file(df: pd.Series, file_name: Path) -> None:
    with open(file_name, "w") as f:
        for _, row in df.items():
            f.write(" ".join([str(ele) for ele in eval(row)]))
            f.write("\n")


def generate_train_test(config: dict[str, Any], n: int | None = None, ratio: float = 0.3) -> None:
    blk_label_dict: dict[str, int] = {}
    blk_label_file = Path(config["input_dir"]) / "anomaly_label.csv"
    blk_df = pd.read_csv(blk_label_file)
    for _, row in tqdm(blk_df.iterrows()):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    seq = pd.read_csv(_log_sequence_file(config))
    seq["Label"] = seq["BlockId"].apply(lambda x: blk_label_dict.get(x))

    normal_seq = seq[seq["Label"] == 0]["EventSequence"]
    normal_seq = normal_seq.sample(frac=1, random_state=20)
    abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
    normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
    train_len = n if n else int(normal_len * ratio)
    print(
        "normal size {0}, abnormal size {1}, training size {2}".format(
            normal_len, abnormal_len, train_len
        )
    )

    output_dir = Path(config["output_dir"])
    train = normal_seq.iloc[:train_len]
    test_normal = normal_seq.iloc[train_len:]
    test_abnormal = abnormal_seq

    df_to_file(train, output_dir / "train")
    df_to_file(test_normal, output_dir / "test_normal")
    df_to_file(test_abnormal, output_dir / "test_abnormal")
    print("generate train test data done")


def run_data_process(config: dict[str, Any]) -> None:
    ensure_output_dirs(config)
    seed_from_config(config)
    parser(config, parser_type=str(config.get("parser_type", "drain")))
    mapping(config)
    hdfs_sampling(config)
    generate_train_test(config, n=int(config.get("train_size", 4855)))


def run_vocab(config: dict[str, Any], vocab_size: int | None, encoding: str, min_freq: int) -> None:
    ensure_output_dirs(config)
    seed_from_config(config)
    options = build_options(config)
    with open(options["train_vocab"], "r", encoding=encoding) as f:
        texts = f.readlines()
    vocab = WordVocab(texts, max_size=vocab_size, min_freq=min_freq)
    print("VOCAB SIZE:", len(vocab))
    print("save vocab in", options["vocab_path"])
    vocab.save_vocab(options["vocab_path"])


def run_train(config: dict[str, Any]) -> None:
    ensure_output_dirs(config)
    seed_from_config(config)
    options = build_options(config)
    print("device", options["device"])
    print("features logkey:{} time: {}\n".format(options["is_logkey"], options["is_time"]))
    print("mask ratio", options["mask_ratio"])
    Trainer(options).train()


def run_predict(config: dict[str, Any], mean: float, std: float) -> None:
    ensure_output_dirs(config)
    seed_from_config(config)
    options = build_options(config)
    options["gaussian_mean"] = mean
    options["gaussian_std"] = std
    print("device", options["device"])
    print("features logkey:{} time: {}\n".format(options["is_logkey"], options["is_time"]))
    print("mask ratio", options["mask_ratio"])
    Predictor(options).predict()
