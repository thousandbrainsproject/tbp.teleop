# tbp.teleop

[![Limited Maintenance](https://img.shields.io/badge/Limited_Maintenance-%E2%9A%A0%EF%B8%8F-red)](#limited-maintenance)

Visualize and teleoperate a [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) experiment in real-time.

> [!NOTE]
>
> Originally created by [@ramyamounir](https://github.com/ramyamounir/) and submitted https://github.com/thousandbrainsproject/tbp.monty/pull/1016.
>
> [@tristanls-tbp](https://github.com/tristanls-tbp/) only converted the tool into the separate repo.

## Installation

The environment for this project is managed with [conda](https://www.anaconda.com/download/success).

### Prerequisites

This project is a tool for [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty). It expects `tbp.monty` to be installed at `../tbp.monty`. If your `tbp.monty` installation is not there, you can update the `environment[_arm64].yaml` file to install from the correct location:

```yaml
  # ...
  - pip:
      - eval-type-backport>=0.3.1
      - opencv-python==4.13.0.92
      - omegaconf>=2.3.0
      - pydantic>=2.10.6
      - typing-extensions
      - -e ../tbp.monty # <-- point to where you have tbp.monty installed
      - -e .[dev]
```

To create the environment, run:

### ARM64 (Apple Silicon) (zsh shell)
```
conda env create -f environment_arm64.yml --subdir=osx-64
conda init zsh
conda activate tbp.teleop
conda config --env --set subdir osx-64
```

### ARM64 (Apple Silicon) (bash shell)
```
conda env create -f environment_arm64.yml --subdir=osx-64
conda init
conda activate tbp.teleop
conda config --env --set subdir osx-64
```

### Intel (zsh shell)
```
conda env create -f environment.yml
conda init zsh
conda activate tbp.teleop
```

### Intel (bash shell)
```
conda env create -f environment.yml
conda init
conda activate tbp.teleop
```

## Experiments

Define your experiments in the `src/tbp/teleop/conf/experiment` directory.
You'll need to pass the `src/tbp/teleop/conf` path as part of your run command
via the Hydra `-cd src/tbp/teleop/conf` option.

After installing the environment, to run an experiment, run:

```bash
python run.py -cd src/tbp/teleop/conf experiment=example
```


## Live Plotter

The live plotter visualizes an experiment in real-time. It is attached to an
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
when the experiment does not already list one in its `defaults`:

```bash
python run.py -cd src/tbp/teleop/conf experiment=example +hooks=interactive
# OR
python run.py -cd src/tbp/teleop/conf experiment=example +hooks=monitor
```

If the experiment already includes a hook in its `defaults` list, drop the `+` to
swap it instead (for example, `hooks=interactive`).

## Development

After installing the environment, you can run the following commands to check your code.

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
