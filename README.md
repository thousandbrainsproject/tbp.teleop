# tbp.teleop

Visualize and teleoperate a [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) experiment in real-time.

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

To run an experiment where episodes are executed in parallel, run:

```bash
python run_parallel.py -cd src/tbp/teleop/conf experiment=example num_parallel=8
```

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
