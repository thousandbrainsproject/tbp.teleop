# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The experiment section of a frame: where the run has got to."""

from __future__ import annotations

from typing import Any, TypedDict


class ExperimentFrame(TypedDict, total=False):
    """Where the run has got to, as the experiment itself counts it.

    Mirrors what Monty's `MontyExperiment.logger_args` gathers for its own loggers,
    which is the experiment's own answer to "where am I", rather than a second opinion
    assembled out here. `total=False` because the experiment reports what it has: a
    `primary_target` only exists for an environment that has one object per episode.

    A frame's own `episode` counts episodes across the whole run, since it is all a step
    hook can see; these count train and eval separately, the way the experiment does.

    Attributes:
        total_train_steps: Steps taken across every training episode so far.
        train_episodes: Training episodes finished so far.
        train_epochs: Training epochs finished so far.
        total_eval_steps: Steps taken across every evaluation episode so far.
        eval_episodes: Evaluation episodes finished so far.
        eval_epochs: Evaluation epochs finished so far.
        episode_seed: The seed this episode's random number generator started from.
        primary_target: The object the episode is about, when the environment has one.
    """

    total_train_steps: int
    train_episodes: int
    train_epochs: int
    total_eval_steps: int
    eval_episodes: int
    eval_epochs: int
    episode_seed: int
    primary_target: dict[str, Any]
