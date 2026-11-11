"""Launch LLaMA-Factory training, prediction and export with traceable run folders.

    python -m llmtrain.run train   CONFIG --data DATA_DIR [--resume RUN_DIR] [key=value ...]
    python -m llmtrain.run predict CONFIG --data DATA_DIR [--adapter RUN_DIR] [key=value ...]
    python -m llmtrain.run export  CONFIG --adapter RUN_DIR [--out EXPORT_DIR] [key=value ...]

train   creates runs/<model>-<YYYYmmdd-HHMMSS>/ and saves next to the checkpoints the exact
        config, the data manifest (dataset revision, seed, split doc_ids, image failures) and
        the command. --resume RUN_DIR re-runs into that folder; LLaMA-Factory then continues
        from its latest checkpoint.
predict creates runs/predict-<training run | model-base>-<timestamp>/, writes
        generated_predictions.jsonl and scores it into metrics.json (llmtrain.scoring).
export  merges the adapter into the bf16 base model (default: <RUN_DIR>/merged).

Extra key=value arguments are passed to llamafactory-cli and override the config file.
"""

import argparse
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from llmtrain.scoring import evaluate_files

RUNS_DIR = Path("runs")


def _config_value(config: Path, key: str) -> str | None:
    """A top-level scalar from a LLaMA-Factory YAML file (no YAML parser needed for these keys)."""
    match = re.search(rf"^{re.escape(key)}:\s*([^\s#]+)", config.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else None


def _model_tag(config: Path) -> str:
    model = _config_value(config, "model_name_or_path") or config.stem
    return model.rstrip("/").split("/")[-1]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _llamafactory(command: str, config: Path, overrides: dict, extra: list[str], record_dir: Path) -> None:
    cli = shutil.which("llamafactory-cli") or "llamafactory-cli"  # which() also finds Windows .cmd shims
    args = [cli, command, str(config), *(f"{k}={v}" for k, v in overrides.items()), *extra]
    with open(record_dir / "commands.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}  {shlex.join(args)}\n")
    print(shlex.join(args), flush=True)
    subprocess.run(args, check=True)


def _record_inputs(run_dir: Path, config: Path, data: Path | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep every config a resumed run was started with, rather than overwriting the first.
    shutil.copy2(config, run_dir / f"config-{_timestamp()}{config.suffix}")
    if data is not None and (data / "manifest.json").is_file():
        shutil.copy2(data / "manifest.json", run_dir / "data_manifest.json")


def _data_overrides(output_dir: Path, data: Path) -> dict:
    # Images in the prepared data are relative to the data folder, so it is also media_dir.
    return {"output_dir": output_dir, "dataset_dir": data, "media_dir": data}


def train(config: Path, data: Path, resume: Path | None, extra: list[str]) -> Path:
    run_dir = resume or RUNS_DIR / f"{_model_tag(config)}-{_timestamp()}"
    if resume is not None and not resume.is_dir():
        raise SystemExit(f"--resume: {resume} is not a run folder")
    _record_inputs(run_dir, config, data)
    overrides = _data_overrides(run_dir, data)
    _llamafactory("train", config, overrides, extra, run_dir)
    print(f"run folder: {run_dir}")
    return run_dir


def predict(config: Path, data: Path, adapter: Path | None, extra: list[str]) -> Path:
    # predict-<training run>-<time> for an adapter, predict-<model>-base-<time> for the base model
    source = adapter.name if adapter else f"{_model_tag(config)}-base"
    out_dir = RUNS_DIR / f"predict-{source}-{_timestamp()}"
    _record_inputs(out_dir, config, data)
    overrides = _data_overrides(out_dir, data)
    if adapter is not None:
        overrides["adapter_name_or_path"] = adapter
    _llamafactory("train", config, overrides, extra, out_dir)  # do_predict runs through `train`

    eval_dataset = next((a.split("=", 1)[1] for a in extra if a.startswith("eval_dataset=")), None)
    eval_dataset = eval_dataset or _config_value(config, "eval_dataset") or "mmqa_test"
    meta = data / f"{eval_dataset}.meta.jsonl"
    result = evaluate_files(out_dir / "generated_predictions.jsonl", meta, out_dir / "metrics.json")
    overall = result["overall"]
    print(f"scored {overall['n']} predictions against {meta}: " + ", ".join(
        f"{k}={v:.3f}" for k, v in overall.items() if isinstance(v, float)
    ))
    print(f"prediction folder: {out_dir}")
    return out_dir


def export(config: Path, adapter: Path, out: Path | None, extra: list[str]) -> Path:
    export_dir = out or adapter / "merged"
    overrides = {"adapter_name_or_path": adapter, "export_dir": export_dir}
    _llamafactory("export", config, overrides, extra, adapter)
    print(f"merged model: {export_dir}")
    return export_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "predict", "export"):
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
        if name in ("train", "predict"):
            p.add_argument("--data", type=Path, required=True, help="output folder of llmtrain.prepare")
        if name == "train":
            p.add_argument("--resume", type=Path, help="existing run folder to continue")
        if name == "predict":
            p.add_argument("--adapter", type=Path, help="run folder whose final adapter to load")
        if name == "export":
            p.add_argument("--adapter", type=Path, required=True)
            p.add_argument("--out", type=Path)
    args, extra = parser.parse_known_args(argv)
    bad = [a for a in extra if "=" not in a]
    if bad:
        parser.error(f"unrecognized arguments: {' '.join(bad)} (overrides must be key=value)")

    if args.command == "train":
        train(args.config, args.data, args.resume, extra)
    elif args.command == "predict":
        predict(args.config, args.data, args.adapter, extra)
    else:
        export(args.config, args.adapter, args.out, extra)


if __name__ == "__main__":
    sys.exit(main())
