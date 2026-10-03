from __future__ import annotations
from solver import format_solution, from_screenshot_state, solve
from visualize import export_html
from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import KMeans


# Game representation:
#   -1 = blank slot
#    0..9 = a detected color
#
# A tube is returned TOP -> BOTTOM because that is how it appears
# in the screenshot. We can reverse this later when implementing
# the pouring/solving logic if we want TOP to mean the movable end.


# Restart-button-relative slot coordinates for the current game's layout.
#
# The reference point is the CENTER of the "Restart" button.
# Each tuple is (dx, dy), in pixels.
#
# There are 3 rows x 4 tubes x 4 slots = 48 positions.
SLOT_OFFSETS = [
    # row 0
    [(-6, -879), (-6, -833), (-6, -777), (-6, -727)],
    [(145, -879), (145, -833), (145, -777), (145, -727)],
    [(295, -879), (295, -833), (295, -777), (295, -727)],
    [(445, -879), (445, -833), (445, -777), (445, -727)],

    # row 1
    [(-6, -599), (-6, -549), (-6, -498), (-6, -447)],
    [(145, -599), (145, -549), (145, -498), (145, -447)],
    [(295, -599), (295, -549), (295, -498), (295, -447)],
    [(445, -599), (445, -549), (445, -498), (445, -447)],

    # row 2
    [(-6, -315), (-6, -264), (-6, -213), (-6, -162)],
    [(145, -315), (145, -264), (145, -213), (145, -162)],
    [(295, -315), (295, -264), (295, -213), (295, -162)],
    [(445, -315), (445, -264), (445, -213), (445, -162)],
]


def find_restart_button(image: np.ndarray) -> tuple[float, float]:
    """Find the center of the cyan Restart button."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Cyan/turquoise UI buttons.
    lower = np.array([85, 120, 150], dtype=np.uint8)
    upper = np.array([105, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # The Restart button is in the lower part of the screenshot.
    mask[:1200] = 0
    mask[1400:] = 0

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area > 1000 and w > 100 and h > 50:
            candidates.append((area, x, y, w, h))

    if not candidates:
        raise RuntimeError("Could not find the Restart button.")

    # The Restart button is the large cyan button on the left.
    _, x, y, w, h = max(candidates, key=lambda c: c[0])
    return x + w / 2, y + h / 2


def get_slot_centers(anchor: tuple[float, float]) -> list[list[tuple[int, int]]]:
    """Convert restart-relative offsets into absolute screenshot coordinates."""
    ax, ay = anchor

    return [
        [(round(ax + dx), round(ay + dy)) for dx, dy in tube]
        for tube in SLOT_OFFSETS
    ]


def sample_slot(image: np.ndarray, center: tuple[int, int], radius: int = 5) -> np.ndarray:
    """
    Sample a small square around the center of a slot.

    Median RGB is used instead of a single pixel so that tiny rendering
    artifacts or compression noise do not affect classification.
    """
    x, y = center

    patch = image[
        max(0, y - radius): y + radius + 1,
        max(0, x - radius): x + radius + 1,
    ]

    if patch.size == 0:
        raise RuntimeError(f"Invalid slot coordinate: {center}")

    # OpenCV stores BGR; convert to RGB.
    rgb = patch[:, :, ::-1]
    return np.median(rgb.reshape(-1, 3), axis=0).astype(np.float32)


def is_blank(rgb: np.ndarray) -> bool:
    """
    Empty slots are very dark. Filled slots have a much higher V/value.

    This deliberately uses brightness rather than saturation so that the
    white color is correctly treated as a real color rather than blank.
    """
    return float(np.max(rgb)) < 100


def build_color_palette(samples: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster the detected filled-slot colors into 10 game colors.

    Returns:
        palette: 10 RGB color centers
        labels:  one label (0..9) per input sample
    """
    filled = np.array([sample for sample in samples if not is_blank(sample)])

    if len(filled) == 0:
        raise RuntimeError("No filled slots were detected.")

    # This game has 10 colors. In a valid board there should be four
    # occurrences of each, so KMeans is a convenient first implementation.
    if len(filled) < 10:
        raise RuntimeError(
            f"Only {len(filled)} filled slots detected; cannot identify 10 colors."
        )

    model = KMeans(n_clusters=10, random_state=0, n_init=10)
    raw_labels = model.fit_predict(filled)
    raw_palette = model.cluster_centers_

    # Give the clusters deterministic IDs by sorting their RGB centers.
    order = np.lexsort(
        (raw_palette[:, 2], raw_palette[:, 1], raw_palette[:, 0])
    )

    palette = raw_palette[order]
    remap = np.empty(10, dtype=np.int32)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id

    labels = remap[raw_labels]
    return palette, labels


def image_to_state(image_path: str | Path):
    """
    Convert a screenshot into a 3x4 board of four-slot tubes.

    Returns:
        state: list of 12 tubes, each containing four values TOP -> BOTTOM
        palette: RGB palette for IDs 0..9
        anchor: detected Restart-button center
        centers: absolute slot centers
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return parse_image(image)


def bytes_to_state(data: bytes):
    """Same as image_to_state(), for an encoded image held in memory."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded file as an image.")

    return parse_image(image)


def parse_image(image: np.ndarray):
    """Parse a decoded BGR screenshot; see image_to_state() for the result."""
    anchor = find_restart_button(image)
    centers = get_slot_centers(anchor)

    # Flatten the 48 screenshot positions while remembering their locations.
    samples = []
    for tube in centers:
        for center in tube:
            samples.append(sample_slot(image, center))

    palette, filled_labels = build_color_palette(samples)

    # Reconstruct the 12 tubes.
    state = []
    filled_index = 0

    for tube in centers:
        parsed_tube = []

        for sample in [sample_slot(image, center) for center in tube]:
            if is_blank(sample):
                parsed_tube.append(-1)
            else:
                parsed_tube.append(int(filled_labels[filled_index]))
                filled_index += 1

        state.append(parsed_tube)

    return state, palette, anchor, centers


def print_state(state, palette, anchor):
    print(f"Restart center: ({anchor[0]:.1f}, {anchor[1]:.1f})")
    print("\nColor IDs (RGB):")
    for i, rgb in enumerate(palette):
        print(f"  {i}: ({round(rgb[0])}, {round(rgb[1])}, {round(rgb[2])})")

    print("\nGame state:")
    for i, tube in enumerate(state, start=1):
        print(f"  Tube {i:2}: {tube}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a Color Puzzle screenshot into game state."
    )
    parser.add_argument("image", help="Path to screenshot")
    parser.add_argument(
        "--html",
        default="solution.html",
        help="Where to write the animated solution page",
    )
    args = parser.parse_args()

    state, palette, anchor, centers = image_to_state(args.image)
    print_state(state, palette, anchor)

    # The parser yields TOP -> BOTTOM with -1 blanks; the solver expects
    # BOTTOM -> TOP with blanks removed.
    solution = solve(from_screenshot_state(state))

    if solution is None:
        print("\nNo solution found.")
    else:
        print()
        print(format_solution(solution))
        path = export_html(solution, palette, args.html)
        print(f"\nAnimated solution written to {path.resolve()}")
