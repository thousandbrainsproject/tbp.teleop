# tbp.teleop

[![Limited Maintenance](https://img.shields.io/badge/Limited_Maintenance-%E2%9A%A0%EF%B8%8F-red)](#limited-maintenance)

Visualize and teleoperate a [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) experiment in real-time.

> [!NOTE]
>
> Originally created by [@ramyamounir](https://github.com/ramyamounir/) and submitted https://github.com/thousandbrainsproject/tbp.monty/pull/1016.
>
> [@tristanls-tbp](https://github.com/tristanls-tbp/) only converted the tool into the separate repo.

## Installation

The environment for this project is managed with [uv](https://docs.astral.sh/uv/getting-started/installation/).

### Prerequisites

This project is a tool for [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty). It expects `tbp.monty` to be checked out at `../tbp.monty`. If your `tbp.monty` checkout is somewhere else, update the path in `pyproject.toml`:

```toml
[tool.uv.sources]
"tbp.monty" = { path = "../tbp.monty", editable = true } # <-- point to where you have tbp.monty
```

### Create the environment

From the root of this repository, run:

```bash
uv sync --extra dev --extra gui --extra simulator_mujoco
```

This creates a `.venv` with a Python version satisfying `requires-python` (3.13 or newer) and installs the locked dependencies from `uv.lock`. The extras are:

- `dev` — formatter, linter, type checker, and test dependencies.
- `gui` — PyQt5, the interactive GUI backend (QtAgg) required by the live plotter.
- `simulator_mujoco` — the MuJoCo simulator used by Monty.

Either activate the environment (`source .venv/bin/activate`) or prefix commands with `uv run`.

## Experiments

To run tbp.teleop with config files in your working tbp.monty checkout, run:

```bash
uv run python run.py experiment=your_experiment_name
```

For example:

```bash
uv run python run.py experiment=supervised_pre_training_cows_small_compositional
```

although with the above example, you will want to introduce the hooks discussed below.

Note that the `src/tbp/teleop/conf` directory is added to Hydra's config search path
automatically by a Hydra plugin (`src/hydra_plugins`). This contains the hook configs (see below),
and can also be used to define your own experiments locally under `src/tbp/teleop/conf/experiment`. For example:

```bash
uv run python run.py experiment=example
```

## Live Plotter

The main purpose of tbp.teleop is to provide a live plotter which visualizes an experiment in real-time. It is attached to an
experiment as a Monty step hook. Two hooks are provided as a Hydra `hooks` config
group under `src/tbp/teleop/conf/hooks`:

- `monitor` — watch the experiment as it runs.
- `interactive` — watch and teleoperate, overriding the agent's action at every step.

Enable a hook by adding it to your experiment's `defaults` list:

```yaml
# @package _global_

defaults:
  - /monty: evidencegraph_exp1000_emin_t3_tot2500
  # ...
  - /hooks: monitor
```

Or select one at run time as an inline override. Use `+hooks=` to add the hook
when the experiment does not already list one in its `defaults`, such as an experiment
from your `tbp.monty` checkout:

```bash
uv run python run.py experiment=supervised_pre_training_cows_small_compositional +hooks=interactive
# OR
uv run python run.py experiment=supervised_pre_training_cows_small_compositional +hooks=monitor
```

If the experiment already includes a hook in its `defaults` list, drop the `+` to
swap it instead. The local `example` experiment lists `/hooks: monitor`, so use:

```bash
uv run python run.py experiment=example hooks=interactive
```

You can also append config overrides as required, for example:

```bash
uv run python run.py experiment=example hooks=interactive logging=basic_warning_monty_runs
```

to run with `logging=basic_warning_monty_runs` rather than defaults which may use Wandb.

## Development

After installing the environment, you can run the following commands to check your code. Prefix them with `uv run` if the environment is not activated.

### Run formatter

```bash
ruff format
```

### Run style checks

```bash
ruff check
```

### Run dependency checks

```bash
deptry .
```

### Run static type checks

```bash
mypy .
```

### Run tests

```bash
pytest
```

## Limited Maintenance

> [!IMPORTANT]
> This repository receives limited maintenance. It is maintained time-permitting at our own discretion. Issues and pull requests may not receive timely responses, and support or updates are not guaranteed. Community contributions are welcome but may be reviewed infrequently.
