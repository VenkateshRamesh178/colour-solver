from __future__ import annotations

"""
A* solver for the color-puzzle / water-sort-style game.

State convention
----------------
A tube is stored BOTTOM -> TOP.

Example:
    (3, 3, 7, 7)

means the bottom two slots are color 3 and the top two slots are color 7.

The screenshot parser currently produces TOP -> BOTTOM values and uses -1
for empty slots. Use `from_screenshot_state()` to convert that representation
before solving.

The solver itself never stores -1 inside a tube. Empty tubes are simply ().
"""

from dataclasses import dataclass
import heapq
import itertools
from typing import Iterable


Color = int
Tube = tuple[Color, ...]
State = tuple[Tube, ...]


@dataclass(frozen=True, slots=True)
class Move:
    """One legal pour from source tube to destination tube."""

    source: int
    destination: int
    amount: int

    def __str__(self) -> str:
        return (
            f"Tube {self.source + 1} -> Tube {self.destination + 1}"
            f" ({self.amount})"
        )


@dataclass(slots=True)
class Solution:
    moves: list[Move]
    states: list[State]
    expanded_states: int


def from_screenshot_state(
    screenshot_state: Iterable[Iterable[int]],
) -> State:
    """
    Convert the parser's TOP -> BOTTOM representation into the solver's
    BOTTOM -> TOP representation.

    -1 means an empty screenshot slot.
    """
    return tuple(
        tuple(color for color in reversed(tuple(tube)) if color != -1)
        for tube in screenshot_state
    )


def validate_state(state: State, capacity: int = 4) -> None:
    """Raise ValueError if the board is structurally invalid."""
    if not state:
        raise ValueError("State must contain at least one tube.")

    counts: dict[int, int] = {}

    for i, tube in enumerate(state, start=1):
        if len(tube) > capacity:
            raise ValueError(
                f"Tube {i} contains {len(tube)} colors; capacity is {capacity}."
            )

        for color in tube:
            if not isinstance(color, int) or color < 0:
                raise ValueError(
                    f"Tube {i} contains invalid color value: {color!r}"
                )
            counts[color] = counts.get(color, 0) + 1

    # In a normal puzzle every color occurs exactly `capacity` times.
    bad = {
        color: count
        for color, count in counts.items()
        if count != capacity
    }
    if bad:
        raise ValueError(
            "Each color must occur exactly "
            f"{capacity} times. Invalid counts: {bad}"
        )


def is_solved(state: State, capacity: int = 4) -> bool:
    """Return True when every non-empty tube contains one complete color."""
    return all(
        not tube or (len(tube) == capacity and len(set(tube)) == 1)
        for tube in state
    )


def bottom_run(tube: Tube) -> tuple[Color, int]:
    """
    Return (bottom_color, run_length).

    Because this game pours from the BOTTOM, the movable end is tube[0].
    """
    color = tube[0]
    amount = 1

    for i in range(1, len(tube)):
        if tube[i] != color:
            break
        amount += 1

    return color, amount


def legal_moves(
    state: State,
    capacity: int = 4,
) -> Iterable[tuple[Move, State]]:
    """
    Generate legal pours.

    A move pours the maximum possible contiguous group from the top of the
    source tube into the destination.

    Important pruning:
    - Never pour from a solved tube.
    - Never pour a color onto a different color.
    - Never pour a complete color group into an empty tube if doing so would
      simply recreate a symmetric state.
    """
    tube_count = len(state)

    for source in range(tube_count):
        src = state[source]

        if not src:
            continue

        # A solved tube never needs to be disturbed.
        if len(src) == capacity and len(set(src)) == 1:
            continue

        color, run = bottom_run(src)

        for destination in range(tube_count):
            if destination == source:
                continue

            dst = state[destination]

            if len(dst) == capacity:
                continue

            # A pour into a non-empty tube is only legal when the movable
            # bottom color of the destination matches.
            if dst and dst[0] != color:
                continue

            free = capacity - len(dst)
            amount = min(run, free)

            # If destination is empty and the entire source top group fits,
            # pouring it there is useful but can be symmetric with another
            # empty destination. Canonicalize equivalent empty destinations
            # below by allowing only the first empty tube.
            if not dst:
                # Moving a single-color tube into an empty one only swaps
                # tube positions; it never makes progress.
                if run == len(src):
                    continue

                first_empty = next(
                    (i for i, tube in enumerate(state) if not tube),
                    destination,
                )
                if destination != first_empty:
                    continue

            # Colors leave the SOURCE from the bottom and enter the
            # DESTINATION at its bottom.
            new_source = src[amount:]

            if dst:
                new_destination = (color,) * amount + dst
            else:
                new_destination = (color,) * amount

            new_state = list(state)
            new_state[source] = new_source
            new_state[destination] = new_destination

            yield Move(source, destination, amount), tuple(new_state)


def color_group_count(state: State) -> int:
    """
    Count contiguous color groups across the whole board.

    Direction does not matter for this heuristic. For example:
        (1, 1, 2, 2, 1)

    contains 3 groups: 1 | 2 | 1.
    """
    groups = 0

    for tube in state:
        if not tube:
            continue

        groups += 1
        for i in range(1, len(tube)):
            if tube[i] != tube[i - 1]:
                groups += 1

    return groups


def heuristic(state: State) -> int:
    """
    Admissible A* heuristic.

    Every solved color must ultimately occupy one contiguous group.
    Therefore:

        groups - number_of_colors

    is a lower bound on the number of moves still required.

    A useful move can reduce the number of groups by at most one, so the
    heuristic can never overestimate the true remaining move count.
    """
    colors = {
        color
        for tube in state
        for color in tube
    }

    if not colors:
        return 0

    return max(0, color_group_count(state) - len(colors))


def _state_key(state: State) -> State:
    """
    Canonicalize equivalent empty tubes.

    Empty tubes are interchangeable. Keeping them as-is would make A*
    search states that are visually/game-theoretically identical.
    """
    return state


def solve(
    initial_state: State | Iterable[Iterable[int]],
    capacity: int = 4,
) -> Solution | None:
    """
    Find a minimum-move solution using A*.

    Returns:
        Solution if one exists, otherwise None.
    """
    state = tuple(tuple(tube) for tube in initial_state)
    validate_state(state, capacity)

    if is_solved(state, capacity):
        return Solution(moves=[], states=[state], expanded_states=0)

    # (f_score, g_score, tie_breaker, state)
    counter = itertools.count()
    start_h = heuristic(state)

    frontier: list[tuple[int, int, int, State]] = [
        (start_h, 0, next(counter), state)
    ]

    # Best known distance from the start to each state.
    g_score: dict[State, int] = {state: 0}

    # Used to reconstruct the winning path.
    parent: dict[State, tuple[State, Move]] = {}

    expanded = 0

    while frontier:
        _, current_g, _, current = heapq.heappop(frontier)

        # A stale heap entry.
        if current_g != g_score.get(current):
            continue

        expanded += 1

        if is_solved(current, capacity):
            return _reconstruct_solution(
                current,
                parent,
                expanded,
            )

        for move, neighbor in legal_moves(current, capacity):
            tentative_g = current_g + 1

            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue

            g_score[neighbor] = tentative_g
            parent[neighbor] = (current, move)

            f_score = tentative_g + heuristic(neighbor)

            heapq.heappush(
                frontier,
                (
                    f_score,
                    tentative_g,
                    next(counter),
                    neighbor,
                ),
            )

    return None


def _reconstruct_solution(
    goal: State,
    parent: dict[State, tuple[State, Move]],
    expanded: int,
) -> Solution:
    moves: list[Move] = []
    states: list[State] = [goal]

    current = goal

    while current in parent:
        previous, move = parent[current]
        moves.append(move)
        states.append(previous)
        current = previous

    moves.reverse()
    states.reverse()

    return Solution(
        moves=moves,
        states=states,
        expanded_states=expanded,
    )


def format_solution(solution: Solution) -> str:
    """Return a human-readable move list."""
    if not solution.moves:
        return "Already solved."

    lines = [
        f"Minimum moves: {len(solution.moves)}",
        f"States expanded: {solution.expanded_states}",
        "",
    ]

    for number, move in enumerate(solution.moves, start=1):
        lines.append(f"{number:3}. {move}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Example state.
    #
    # IMPORTANT:
    # This is BOTTOM -> TOP. The FIRST value is the movable bottom.
    # Replace it with the state produced by image_to_state.py after calling
    # from_screenshot_state().
    #
    # Example:
    # screenshot_state = [
    #     [3, 3, 0, 0],
    #     ...
    # ]
    # state = from_screenshot_state(screenshot_state)

    example_state: State = (
        (0, 0),
        (1, 1, 2, 2),
        (3, 3, 4, 4),
        (5, 5, 6, 6),
        (7, 7, 8, 8),
        (9, 9),
        (),
        (),
    )

    # The example above is intentionally not a complete 10-color puzzle,
    # so don't run it as-is. This block is here as API documentation.
    print("Import this module and call solve(state).")
