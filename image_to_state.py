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


# Tubes are found from their outlines, so no fixed layout is assumed:
# any phone, resolution, aspect ratio or crop works as long as the
# whole board is visible.
SLOTS_PER_TUBE = 4

# Slot centers along a tube, as a fraction of the tube's outline height
# measured from its center. Blocks are evenly spaced, 0.2125 heights apart.
SLOT_SPACING = 0.2125

# Gray levels separating the tube outlines from the board background.
# Several are tried because themes and image compression shift both.
OUTLINE_THRESHOLDS = (40, 50, 32, 60, 25, 75)


def find_tube_candidates(gray: np.ndarray, threshold: int) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (x, y, w, h) of tall, rounded-rectangle outlines."""
    mask = (gray >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_side = max(gray.shape) / 100
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_side or not 1.4 <= h / w <= 2.6:
            continue
        # A tube outline is a solid, nearly rectangular region.
        if cv2.contourArea(contour) < 0.8 * w * h:
            continue
        boxes.append((x, y, w, h))
    return boxes


def largest_uniform_group(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """
    The largest set of same-sized, non-overlapping boxes: the tubes.

    Each tube produces several nested outlines (outer frame, inner frame),
    so for every size the outermost box per location is kept.
    """
    best = []
    for _, _, ref_w, ref_h in boxes:
        similar = [
            b for b in boxes
            if abs(b[2] - ref_w) <= 0.08 * ref_w and abs(b[3] - ref_h) <= 0.08 * ref_h
        ]
        similar.sort(key=lambda b: b[2] * b[3], reverse=True)

        group = []
        for box in similar:
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            if all(abs(cx - (g[0] + g[2] / 2)) > g[2] / 2 or abs(cy - (g[1] + g[3] / 2)) > g[3] / 2 for g in group):
                group.append(box)

        if len(group) > len(best):
            best = group
    return best


def find_tubes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find the tubes' bounding boxes, ordered row by row, left to right."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    tubes = []
    for threshold in OUTLINE_THRESHOLDS:
        group = largest_uniform_group(find_tube_candidates(gray, threshold))
        if len(group) > len(tubes):
            tubes = group

    if len(tubes) < 3:
        raise RuntimeError(
            "Could not find the tubes. Make sure the whole board is visible."
        )

    # Group into rows: tubes whose centers are within half a tube height.
    tubes.sort(key=lambda b: b[1] + b[3] / 2)
    rows = []
    for box in tubes:
        cy = box[1] + box[3] / 2
        if rows and abs(cy - rows[-1][0]) < box[3] / 2:
            rows[-1][1].append(box)
        else:
            rows.append((cy, [box]))

    return [box for _, row in rows for box in sorted(row, key=lambda b: b[0])]


def get_slot_centers(tubes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int]]]:
    """Slot centers of every tube, TOP -> BOTTOM."""
    centers = []
    for x, y, w, h in tubes:
        cx, cy = x + w / 2, y + h / 2
        offsets = [(i - (SLOTS_PER_TUBE - 1) / 2) * SLOT_SPACING for i in range(SLOTS_PER_TUBE)]
        centers.append([(round(cx), round(cy + f * h)) for f in offsets])
    return centers


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
    Cluster the detected filled-slot colors into the game's colors.

    Every color fills exactly one tube, so there are
    (filled slots / SLOTS_PER_TUBE) of them: 10 in the usual board.

    Returns:
        palette: one RGB center per color
        labels:  one color ID per filled sample
    """
    filled = np.array([sample for sample in samples if not is_blank(sample)])

    if len(filled) == 0:
        raise RuntimeError("No filled slots were detected.")

    n_colors, remainder = divmod(len(filled), SLOTS_PER_TUBE)
    if remainder or n_colors < 2:
        raise RuntimeError(
            f"Detected {len(filled)} filled slots, which is not a whole number "
            f"of {SLOTS_PER_TUBE}-block colors. Use a screenshot of the "
            "unplayed board."
        )

    model = KMeans(n_clusters=n_colors, random_state=0, n_init=10)
    raw_labels = model.fit_predict(filled)
    raw_palette = model.cluster_centers_

    # Give the clusters deterministic IDs by sorting their RGB centers.
    order = np.lexsort(
        (raw_palette[:, 2], raw_palette[:, 1], raw_palette[:, 0])
    )

    palette = raw_palette[order]
    remap = np.empty(n_colors, dtype=np.int32)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id

    labels = remap[raw_labels]
    return palette, labels


def image_to_state(image_path: str | Path):
    """
    Convert a screenshot into a list of four-slot tubes.

    Returns:
        state: one list per tube, four values TOP -> BOTTOM, row by row
        palette: RGB palette, indexed by color ID
        tubes: detected tube bounding boxes (x, y, w, h)
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
    height, width = image.shape[:2]
    try:
        tubes = find_tubes(image)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} (image is {width}x{height} pixels)") from None
    centers = get_slot_centers(tubes)

    # Sample a patch about a tenth of the block width, whatever the scale.
    radius = max(1, round(min(w for _, _, w, _ in tubes) * 0.04))
    sampled = [[sample_slot(image, center, radius) for center in tube] for tube in centers]

    palette, filled_labels = build_color_palette([s for tube in sampled for s in tube])

    # Reconstruct the tubes.
    state = []
    filled_index = 0

    for tube in sampled:
        parsed_tube = []

        for sample in tube:
            if is_blank(sample):
                parsed_tube.append(-1)
            else:
                parsed_tube.append(int(filled_labels[filled_index]))
                filled_index += 1

        state.append(parsed_tube)

    return state, palette, tubes, centers


def print_state(state, palette, tubes):
    print(f"Found {len(tubes)} tubes, each about {tubes[0][2]}x{tubes[0][3]} pixels")
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

    state, palette, tubes, centers = image_to_state(args.image)
    print_state(state, palette, tubes)

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
