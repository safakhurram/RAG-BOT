from src.ocr_layout import order_boxes, to_boxes


def quad(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_two_columns_are_read_column_by_column_under_a_full_width_title():
    items = [(quad(100, 50, 900, 80), "PROSPECTUS TITLE")]
    for i in range(5):
        y = 120 + i * 40
        items.append((quad(100, y, 700, y + 30), f"left line {i} with some text"))
        items.append((quad(850, y, 1500, y + 30), f"right line {i} with some text"))

    lines = order_boxes(to_boxes(items), page_w=1650)

    assert lines[0] == "PROSPECTUS TITLE"
    assert lines[1:6] == [f"left line {i} with some text" for i in range(5)]
    assert lines[6:] == [f"right line {i} with some text" for i in range(5)]


def test_single_column_page_keeps_line_order():
    items = [(quad(100, 100 + i * 40, 1500, 130 + i * 40), f"line {i}") for i in range(4)]
    assert order_boxes(to_boxes(items), page_w=1650) == [f"line {i}" for i in range(4)]


def test_table_rows_stay_together():
    items = []
    rows = [("Categories SF-1 to SF-5", "Rs. 855,000/-"), ("Categories SF-6 to SF-11", "Rs. 1,065,000/-"),
            ("For Foreigners", "Rs. 336,630/-"), ("BS (CT) Programmes", "Rs. 56,390/-")]
    for i, (label, amount) in enumerate(rows):
        y = 100 + i * 40
        items.append((quad(100, y, 600, y + 30), label))
        items.append((quad(1100, y, 1350, y + 30), amount))

    lines = order_boxes(to_boxes(items), page_w=1650)

    assert lines == [f"{label} | {amount}" for label, amount in rows]
