"""Turn raw OCR boxes into text in the order a human reads the page.

OCR engines return text lines sorted top-to-bottom across the WHOLE page.
On a two-column page (most of the NED prospectus) that interleaves the
columns line by line:

    8.1 FEE AND CHARGES              <- left column
    8.2 CHARGES UNDER SELF-FINANCE   <- right column
    a) Admission Fee ...             <- left
    a) Self-Finance Fee must be ...  <- right

so a fee amount ends up next to a label from the other column and the bot
"reads" facts that are not in the PDF. This module uses the box coordinates
to find column gutters (vertical strips with no text) and outputs each column
top-to-bottom before moving right. Lines that run across a gutter (titles,
full-width paragraphs) split the page into horizontal bands, so a page with a
full-width heading above two columns is also read correctly.

Table rows (short cells sitting side by side on the same line) are kept as
rows joined with " | " instead of being split into columns, so "SF-1 to SF-5 |
Rs. 855,000/-" stays together.
"""

from dataclasses import dataclass
from statistics import median


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0


def to_boxes(items) -> list[Box]:
    """items: iterable of (quad, text) where quad is 4 [x, y] points."""
    boxes = []
    for quad, text in items:
        text = (text or "").strip()
        if not text:
            continue
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        boxes.append(Box(min(xs), min(ys), max(xs), max(ys), text))
    return boxes


def _find_gutters(boxes: list[Box], page_w: float) -> list[tuple[float, float]]:
    """Vertical strips in the middle of the page crossed by only a few boxes
    (titles, full-width lines), with real text on both sides."""
    if len(boxes) < 6:
        return []
    width = int(page_w) + 1
    count = [0] * width
    for b in boxes:
        for x in range(max(0, int(b.x0)), min(width, int(b.x1) + 1)):
            count[x] += 1
    allowed = max(1, int(0.12 * len(boxes)))  # boxes allowed to cross a gutter

    min_gap = 0.015 * page_w
    lo, hi = 0.15 * page_w, 0.85 * page_w
    candidates, start = [], None
    for x in range(width + 1):
        free = x < width and count[x] <= allowed
        if free and start is None:
            start = x
        elif not free and start is not None:
            if x - start >= min_gap and start > lo and x < hi:
                candidates.append((start, x))
            start = None

    # The prospectus uses at most two page columns, so keep the ONE strip in
    # the middle of the page that the fewest lines cross. Other white strips
    # (between a course title and its credit-hour cells, or between a list
    # number and a name) are inside a column, not between columns.
    best, best_cost = None, None
    for a, g in candidates:
        if not (0.3 * page_w <= (a + g) / 2 <= 0.7 * page_w):
            continue
        left = sum(1 for b in boxes if b.x1 <= a)
        right = sum(1 for b in boxes if b.x0 >= g)
        if left < 3 or right < 3:
            continue
        cost = (min(count[a:g]), -(g - a))  # fewest crossings, then widest
        if best_cost is None or cost < best_cost:
            best, best_cost = (a, g), cost
    if best is None:
        return []
    # narrow the gutter to its least-crossed part
    a, g = best
    low = min(count[a:g])
    xs = [x for x in range(a, g) if count[x] == low]
    return [(xs[0], xs[-1] + 1)]


def _group_lines(boxes: list[Box]) -> list[list[Box]]:
    """Group boxes that sit on the same visual line, top to bottom."""
    lines: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        if lines:
            last = lines[-1]
            ref = median(x.cy for x in last)
            ref_h = median(x.h for x in last)
            if abs(b.cy - ref) < 0.5 * min(ref_h, b.h):
                last.append(b)
                continue
        lines.append([b])
    return [sorted(line, key=lambda b: b.x0) for line in lines]


def _line_text(line: list[Box], sep: str = " ") -> str:
    return sep.join(b.text for b in line)


def _is_table(columns: list[list[Box]]) -> bool:
    """Side-by-side short cells on the same rows => a table, not text columns."""
    cols = [c for c in columns if c]
    if len(cols) < 2:
        return False
    all_boxes = [b for c in cols for b in c]
    aligned = 0
    for i, col in enumerate(cols):
        others = [b for j, c in enumerate(cols) if j != i for b in c]
        for b in col:
            if any(abs(b.cy - o.cy) < 0.5 * min(b.h, o.h) for o in others):
                aligned += 1
    short_col = any(median(len(b.text) for b in c) <= 14 for c in cols)
    return aligned / len(all_boxes) >= 0.6 and short_col


def _side_by_side_tables(columns: list[list[Box]]) -> bool:
    """Every column is itself a multi-cell table (several boxes per line)."""
    cols = [c for c in columns if c]
    if len(cols) < 2:
        return False
    for col in cols:
        lines = _group_lines(col)
        if sum(len(l) for l in lines) / len(lines) < 2:
            return False
    return True


def order_boxes(boxes: list[Box], page_w: float) -> list[str]:
    """Return the page's text lines in reading order."""
    if not boxes:
        return []
    gutters = _find_gutters(boxes, page_w)
    if not gutters:
        return [_line_text(line) for line in _group_lines(boxes)]

    bounds = [0.0] + [(a + b) / 2 for a, b in gutters] + [float("inf")]

    def spans(b: Box) -> bool:
        return any(b.x0 < a and b.x1 > g for a, g in gutters)

    def column_of(b: Box) -> int:
        cx = (b.x0 + b.x1) / 2
        for i in range(len(bounds) - 1):
            if bounds[i] <= cx < bounds[i + 1]:
                return i
        return len(bounds) - 2

    out: list[str] = []
    band: list[Box] = []

    def flush_band() -> None:
        if not band:
            return
        columns: list[list[Box]] = [[] for _ in range(len(bounds) - 1)]
        for b in band:
            columns[column_of(b)].append(b)
        if _is_table(columns) and not _side_by_side_tables(columns):
            # label | value rows (e.g. "SF-1 to SF-5 | Rs. 855,000/-")
            out.extend(_line_text(line, " | ") for line in _group_lines(band))
        elif _is_table(columns):
            # two complete tables next to each other (e.g. Fall | Spring
            # semester): read the left table fully, then the right one
            for col in columns:
                out.extend(_line_text(line, " | ") for line in _group_lines(col))
        else:
            for col in columns:
                out.extend(_line_text(line) for line in _group_lines(col))
        band.clear()

    # A line that crosses the gutter is read in place, together with the boxes
    # right next to it on the same line (e.g. the section number "2.2.2" in
    # front of a long heading), so they are not separated.
    attached: dict[int, list[Box]] = {}
    taken: set[int] = set()
    for sb in boxes:
        if not spans(sb):
            continue
        group = [sb]
        grew = True
        while grew:
            grew = False
            lo, hi = min(x.x0 for x in group), max(x.x1 for x in group)
            for b in boxes:
                if id(b) in taken or b in group or spans(b):
                    continue
                same_line = abs(b.cy - sb.cy) < 0.5 * min(b.h, sb.h)
                near = b.x1 >= lo - 0.03 * page_w and b.x0 <= hi + 0.03 * page_w
                if same_line and near:
                    group.append(b)
                    taken.add(id(b))
                    grew = True
        attached[id(sb)] = sorted(group, key=lambda x: x.x0)

    for b in sorted(boxes, key=lambda b: b.cy):
        if id(b) in taken:
            continue
        if spans(b):
            flush_band()
            out.append(_line_text(attached[id(b)]))
        else:
            band.append(b)
    flush_band()
    return out
