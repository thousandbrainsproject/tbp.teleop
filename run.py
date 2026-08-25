# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Entrypoint for running an experiment."""

import sys

# `--attention_vis` is a convenience alias for the Hydra override enabling the
# attention visualization layout on the teleop plotter. It requires a teleop hook
# (e.g. `+hooks=monitor`) to be part of the run. Substituted in place so the
# override stays ordered with the other positional overrides.
if "--attention_vis" in sys.argv:
    sys.argv[sys.argv.index("--attention_vis")] = (
        "experiment.config.step_hook.plotter.attention_vis=true"
    )

from tbp.monty.frameworks.run_env import setup_env

setup_env()

from tbp.monty.frameworks.run import main  # noqa: E402

if __name__ == "__main__":
    main()
