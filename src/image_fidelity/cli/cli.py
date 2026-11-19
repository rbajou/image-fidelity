"""``imfid`` command line: run an experiment or a task (restore, benchmark, train)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from image_fidelity.tasks import dispatch

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
TASKS = ("restore", "segment", "benchmark", "train", "plot", "download")
# Leading task names whose Hydra task group has a different name; `train` keeps selecting restoration training.
TASK_GROUPS = {"train": "train_restoration"}
HYDRA_FLAGS_WITH_VALUE = {"--config-path", "-cp", "--config-name", "-cn", "--config-dir", "-cd"}


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base=None)
def _run(config: DictConfig) -> None:
    dispatch(config)


def _is_sweep(argument: str) -> bool:
    key, sep, value = argument.partition("=")
    return bool(sep) and "," in value and not key.startswith(("hydra", "+hydra"))


def translate_arguments(arguments: list[str]) -> list[str]:
    """Turn ``imfid [task] overrides...`` into Hydra arguments.

    A leading task name becomes ``task=<name>``. For ``benchmark``, a comma-separated
    ``method=a,b,c`` selects several methods compared in a single run; any other
    comma-separated override starts a Hydra multirun sweep.
    """
    arguments = list(arguments)
    task = None
    if arguments and arguments[0] in TASKS:
        task = arguments.pop(0)
        arguments.insert(0, f"task={TASK_GROUPS.get(task, task)}")

    if task == "benchmark":
        translated = []
        for argument in arguments:
            if argument.startswith("method=") and argument.count("=") == 1:
                names = [name for name in argument.split("=", 1)[1].split(",") if name]
                translated.extend(f"+method@task.methods.{name}={name}" for name in names)
            else:
                translated.append(argument)
        arguments = translated

    if task == "download":
        # `imfid download <collection> <id>...` -> task.collection / task.datasets overrides.
        positionals = [a for a in arguments[1:] if "=" not in a and not a.startswith("-")]
        if positionals:
            arguments = [a for a in arguments if a not in positionals]
            ids = ",".join(f"'{name}'" for name in positionals[1:])
            arguments += [f"task.collection={positionals[0]}", f"task.datasets=[{ids}]"]

    if "-m" not in arguments and "--multirun" not in arguments and any(map(_is_sweep, arguments)):
        arguments.insert(0, "--multirun")
    return arguments


def _run_compatibly() -> None:
    if sys.version_info < (3, 14):
        _run()
        return

    original_check_help = argparse.ArgumentParser._check_help

    def check_help_compatibly(parser: argparse.ArgumentParser, action: argparse.Action) -> None:
        if type(action.help).__name__ == "LazyCompletionHelp":
            action.help = repr(action.help)
        original_check_help(parser, action)

    argparse.ArgumentParser._check_help = check_help_compatibly
    try:
        _run()
    finally:
        argparse.ArgumentParser._check_help = original_check_help


def main(argv: list[str] | None = None) -> None:
    program, *arguments = sys.argv if argv is None else ["imfid", *argv]
    sys.argv = [program, *translate_arguments(arguments)]
    _run_compatibly()
