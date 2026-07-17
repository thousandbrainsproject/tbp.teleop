# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Command a running experiment from another process.

The far side of the control channel. Like `watch`, it imports no Monty, so it can run
in an environment built for whatever a driver wants rather than for whatever habitat
tolerates. It sends a structured command and prints where the run stands after it:

    python -m tbp.teleop.drive step                    # advance one step
    python -m tbp.teleop.drive step --action turn_left --param rotation_degrees=5
    python -m tbp.teleop.drive step --goal 0,1.5,0     # a place to look at
    python -m tbp.teleop.drive auto                    # keep stepping to the end
    python -m tbp.teleop.drive continuous --interval 0.5   # let it run itself
    python -m tbp.teleop.drive stepmode                # go back to stepping
    python -m tbp.teleop.drive quit
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from tbp.teleop.commands import (
    QuitCommand,
    RunMode,
    SetRunModeCommand,
    StepCommand,
)
from tbp.teleop.frames import ActionFrame, GoalFrame, Pursue
from tbp.teleop.wire import DEFAULT_CONTROL_ENDPOINT, CommandClient

if TYPE_CHECKING:
    from tbp.teleop.commands import Command

# Long enough for the experiment to reach its hook between steps, even while habitat is
# rendering, but finite: when a run ends it stops answering, and the driver must notice
# rather than wait for ever.
DEFAULT_TIMEOUT = 30.0

# Habitat's single agent, which a spelled-out action must name. The driver does not see
# frames on this channel, so it cannot read the agent from one; every experiment we
# drive has just the one.
AGENT = "agent_id_0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: The arguments to parse, or `None` to read `sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[3:]),
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_CONTROL_ENDPOINT,
        help=f"the endpoint to command on (default: {DEFAULT_CONTROL_ENDPOINT})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"seconds to wait for each acknowledgement (default: {DEFAULT_TIMEOUT})",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    step = commands.add_parser("step", help="take one step, in step mode")
    step.add_argument("--action", help="an action to run, e.g. turn_left")
    step.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a parameter of --action, repeatable, e.g. --param distance=0.01",
    )
    step.add_argument("--goal", metavar="X,Y,Z", help="a place to send the agent")
    step.add_argument(
        "--sender-type",
        choices=list(Pursue),
        type=Pursue,
        default=Pursue.LOOK_AT,
        help="look at the goal (SM), the default, or jump to it (GSG)",
    )

    auto = commands.add_parser("auto", help="keep stepping until the run ends")
    auto.add_argument(
        "--steps", type=int, default=None, help="stop after this many steps"
    )

    continuous = commands.add_parser("continuous", help="let the experiment run itself")
    continuous.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="seconds between steps, 0 for full speed (default: 0.0)",
    )

    commands.add_parser("stepmode", help="return to stepping on command")
    commands.add_parser("quit", help="end the run")
    return parser.parse_args(argv)


def parse_params(pairs: list[str]) -> dict[str, Any]:
    """Read `NAME=VALUE` parameters, as numbers where they look like numbers.

    Args:
        pairs: The `NAME=VALUE` strings.

    Returns:
        The parameters.
    """
    return {
        name: _number(value)
        for name, _, value in (pair.partition("=") for pair in pairs)
    }


def _number(text: str) -> Any:  # noqa: ANN401
    """Read a parameter's value.

    Args:
        text: The value as it was typed.

    Returns:
        The value as an `int` or a `float` where it reads as one, else as it was typed.
    """
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _vector(text: str) -> list[float]:
    """Read an `X,Y,Z` vector.

    Args:
        text: The comma-separated numbers.

    Returns:
        The numbers.
    """
    return [float(part) for part in text.split(",")]


def step_command(args: argparse.Namespace) -> Command:
    """Build the `STEP` command the arguments describe.

    Args:
        args: The parsed command line.

    Returns:
        A step carrying the driver's action, goal, or neither.
    """
    if args.action:
        action = ActionFrame(
            action_name=args.action, agent_id=AGENT, params=parse_params(args.param)
        )
        return StepCommand(actions=[action])
    if args.goal:
        goal = GoalFrame(
            location=_vector(args.goal),
            sender_type=args.sender_type,
        )
        return StepCommand(goals=[goal])
    return StepCommand()


def main(argv: list[str] | None = None) -> None:
    """Send the command the arguments describe, and report the result.

    Args:
        argv: The arguments to parse, or `None` to read `sys.argv`.
    """
    args = parse_args(argv)
    client = CommandClient(args.endpoint, timeout=args.timeout)
    try:
        if args.command == "auto":
            _run_auto(client, args.steps)
        else:
            _run_once(client, args)
    finally:
        client.close()


def _run_once(client: CommandClient, args: argparse.Namespace) -> None:
    """Send a single command and print where the run stands.

    Args:
        client: The control channel.
        args: The parsed command line.
    """
    command: Command = {
        "step": lambda: step_command(args),
        "continuous": lambda: SetRunModeCommand(
            run_mode=RunMode.CONTINUOUS, interval=args.interval
        ),
        "stepmode": lambda: SetRunModeCommand(run_mode=RunMode.STEP),
        "quit": QuitCommand,
    }[args.command]()

    result = client.send(command)
    _report(result)


def _run_auto(client: CommandClient, steps: int | None) -> None:
    """Step until the run ends, or until `steps` have been taken.

    Args:
        client: The control channel.
        steps: The most steps to take, or `None` for no limit.
    """
    taken = 0
    while steps is None or taken < steps:
        result = client.send(StepCommand())
        if result is None:
            print("the experiment stopped answering")  # noqa: T201
            return
        taken += 1
        if result.stopped:
            break
        print(  # noqa: T201
            f"episode {result.episode} step {result.step}", flush=True
        )
    print(f"took {taken} steps")  # noqa: T201


def _report(result: Any) -> None:  # noqa: ANN401
    """Print a command's acknowledgement.

    Args:
        result: The acknowledgement, or `None` when none came.
    """
    if result is None:
        print("the experiment did not answer")  # noqa: T201
        return
    where = f"episode {result.episode} step {result.step}"
    print(f"mode {result.run_mode}, {where}, stopped={result.stopped}")  # noqa: T201


if __name__ == "__main__":
    main()
