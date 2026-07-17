# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Watch a running experiment from another process.

The entry point for the far side of the wire. It imports no Monty, so it can run in an
environment built for whatever the viewer wants rather than for whatever habitat
tolerates.

    python -m tbp.teleop.watch --timeout 30
"""

from __future__ import annotations

import argparse

from tbp.teleop.consumers import ObservationSaver
from tbp.teleop.wire import DEFAULT_ENDPOINT, subscribe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: The arguments to parse, or `None` to read `sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"the endpoint to subscribe to (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--save-path",
        default="watched",
        help="directory to draw into, under the repo's local/ (default: watched)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="seconds of silence before giving up (default: wait for ever)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Subscribe to an experiment and draw every frame it publishes.

    Args:
        argv: The arguments to parse, or `None` to read `sys.argv`.
    """
    args = parse_args(argv)
    consumer = ObservationSaver(args.save_path)
    # This is a console program, so its output is its interface rather than debugging
    # left behind, hence the prints.
    print(f"watching {args.endpoint}, drawing into {consumer.save_path}")  # noqa: T201

    watched = 0
    try:
        for frame in subscribe(args.endpoint, timeout=args.timeout):
            consumer(frame)
            watched += 1
            print(f"episode {frame.episode} step {frame.step}", flush=True)  # noqa: T201
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

    print(f"drew {watched} frames")  # noqa: T201


if __name__ == "__main__":
    main()
