from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from litmus_ir import LitmusCaseIR


WIDTH = 1600
HEIGHT = 1100
FONT_REG = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
FONT_MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")

COLORS = {
    "bg": (250, 250, 250),
    "panel": (255, 255, 255),
    "border": (205, 205, 207),
    "text": (35, 37, 42),
    "muted": (88, 92, 100),
    "load": (221, 238, 216),
    "store": (255, 242, 204),
    "vector": (232, 246, 244),
    "cmo": (238, 231, 247),
    "fence": (227, 238, 249),
    "dep": (255, 236, 233),
    "local": (92, 92, 92),
    "rel": (216, 56, 48),
    "rel_bg": (255, 242, 241),
    "blue": (54, 107, 170),
    "green": (62, 135, 88),
    "spine": (176, 181, 189),
}

# Relation kind -> (line color, accent for chip border/text)
REL_STYLE = {
    "rfe": (49, 140, 99),    # reads-from external: green
    "fre": (210, 67, 53),    # from-read external: red
    "co": (61, 105, 173),    # coherence (Wse): blue
    "obs": (120, 124, 132),  # observation only: gray
}

# Vertical band reserved above the hart panels for over-the-top arches.
PANEL_TOP = 230
PANEL_BOTTOM = 748
PANEL_LEFT = 70
PANEL_RIGHT = 1530
PANEL_GAP = 60
LANE_TOP = 150
LANE_BASE = 212
LANE_STEP = 17


@dataclass(frozen=True)
class DiagramResult:
    png_path: Path
    json_path: Path
    summary: dict[str, Any]


def render_diagram(case_ir: LitmusCaseIR, solver: dict[str, Any] | None, out_dir: Path) -> DiagramResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{case_ir.name}.diagram.png"
    json_path = out_dir / f"{case_ir.name}.diagram.json"
    summary = diagram_summary(case_ir, solver, png_path)
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    _draw_header(draw, fonts, case_ir, solver)
    event_boxes, columns = _draw_harts(draw, fonts, case_ir)
    _draw_relations(draw, fonts, case_ir, event_boxes, columns)
    _draw_cycle(draw, fonts, case_ir)
    _draw_footer(draw, fonts, case_ir, solver)
    image.save(png_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DiagramResult(png_path, json_path, summary)


def diagram_summary(case_ir: LitmusCaseIR, solver: dict[str, Any] | None, png_path: Path | None = None) -> dict[str, Any]:
    return {
        "schema": "litmus-link.diagram.v1",
        "name": case_ir.name,
        "png": str(png_path) if png_path else "",
        "cycle": case_ir.cycle,
        "exists": case_ir.exists,
        "verdict": (solver or {}).get("verdict", "unchecked"),
        "solver_status": (solver or {}).get("status", "not_applicable"),
        "harts": case_ir.hart_names(),
        "events": [event.to_json() for event in case_ir.events()],
        "relations": [relation.to_json() for relation in case_ir.relations],
    }


def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    return {
        "title": _font(34, True),
        "h1": _font(24, True),
        "h2": _font(20, True),
        "body": _font(18),
        "small": _font(15),
        "small_b": _font(15, True),
        "mono": ImageFont.truetype(str(FONT_MONO), 15),
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REG if FONT_REG.exists() else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size)


def _draw_header(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.FreeTypeFont], case_ir: LitmusCaseIR, solver: dict[str, Any] | None) -> None:
    _round(draw, (40, 30, 1560, 135), COLORS["panel"], COLORS["border"], 14)
    draw.text((65, 52), case_ir.display_name, fill=COLORS["text"], font=fonts["title"])
    draw.text((65, 96), case_ir.description or case_ir.name, fill=COLORS["muted"], font=fonts["body"])
    verdict = (solver or {}).get("verdict", "unchecked")
    badge = _solver_badge(solver)
    _round(draw, (1230, 55, 1515, 100), _verdict_color(verdict), COLORS["border"], 10)
    draw.text((1248, 68), badge, fill=COLORS["text"], font=fonts["small_b"])


def _draw_harts(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.FreeTypeFont],
    case_ir: LitmusCaseIR,
) -> tuple[dict[str, tuple[int, int, int, int]], list[tuple[int, int]]]:
    hart_count = max(len(case_ir.harts), 1)
    col_width = int((PANEL_RIGHT - PANEL_LEFT - PANEL_GAP * (hart_count - 1)) / hart_count)
    box_h = 64
    content_top = PANEL_TOP + 58
    content_bottom = PANEL_BOTTOM - 24
    content_height = content_bottom - content_top
    po_labels = _local_label_map(case_ir)
    event_boxes: dict[str, tuple[int, int, int, int]] = {}
    columns: list[tuple[int, int]] = []
    for hart_index, events in enumerate(case_ir.harts):
        x0 = PANEL_LEFT + hart_index * (col_width + PANEL_GAP)
        x1 = x0 + col_width
        columns.append((x0, x1))
        _round(draw, (x0, PANEL_TOP, x1, PANEL_BOTTOM), COLORS["panel"], COLORS["border"], 12)
        draw.text((x0 + 18, PANEL_TOP + 14), f"P{hart_index}", fill=COLORS["blue"], font=fonts["h1"])
        n = max(len(events), 1)
        pitch = min(150, content_height / n)
        block_height = (n - 1) * pitch + box_h
        y = content_top + max(0, (content_height - block_height) / 2)
        previous_event = None
        previous_bottom = None
        for event in events:
            box = (x0 + 18, int(y), x1 - 18, int(y) + box_h)
            _round(draw, box, _event_color(event.kind), COLORS["border"], 8)
            label = f"{event.event_id}: {event.instruction}"
            _fit_text(draw, label, (box[0] + 12, box[1] + 9, box[2] - 12, box[3] - 28), fonts["mono"], COLORS["text"])
            detail = "  ".join(part for part in [event.kind, event.location, event.role] if part)
            draw.text((box[0] + 12, box[3] - 23), detail, fill=COLORS["muted"], font=fonts["small"])
            mid_x = (box[0] + box[2]) // 2
            if previous_event is not None and previous_bottom is not None:
                key = (previous_event, event.event_id)
                _draw_spine(draw, fonts, (mid_x, previous_bottom), (mid_x, box[1]), po_labels.get(key, "po"))
            previous_event = event.event_id
            previous_bottom = box[3]
            event_boxes[event.event_id] = box
            y += pitch
    return event_boxes, columns


def _draw_spine(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.FreeTypeFont],
    start: tuple[int, int],
    end: tuple[int, int],
    label: str,
) -> None:
    draw.line([start, end], fill=COLORS["spine"], width=2)
    _arrowhead(draw, start, end, COLORS["spine"], size=10)
    mid_y = (start[1] + end[1]) // 2
    draw.text((start[0] + 10, mid_y - 8), label, fill=COLORS["muted"], font=fonts["small"])


def _local_label_map(case_ir: LitmusCaseIR) -> dict[tuple[str, str], str]:
    return {
        (relation.src, relation.dst): relation.label
        for relation in case_ir.relations
        if relation.local
    }


def _draw_relations(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.FreeTypeFont],
    case_ir: LitmusCaseIR,
    boxes: dict[str, tuple[int, int, int, int]],
    columns: list[tuple[int, int]],
) -> None:
    for routed in _route_relations(case_ir, boxes, columns):
        color, accent = _relation_style(routed["kind"])
        _draw_curve(draw, fonts, routed["points"], color, accent, routed["label"], routed["slot"])


def _route_relations(
    case_ir: LitmusCaseIR,
    boxes: dict[str, tuple[int, int, int, int]],
    columns: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Pure geometry: turn cross-hart relations into smoothed point lists.

    Adjacent harts get side curves through the column gap; distant harts get
    nested over-the-top arches. Multiple relations touching the same box edge
    are spread to distinct attachment points so they never run parallel.
    """
    events = case_ir.event_map()
    items: list[dict[str, Any]] = []
    for relation in case_ir.relations:
        if relation.local:
            continue
        src = boxes.get(relation.src)
        dst = boxes.get(relation.dst)
        if src is None or dst is None:
            continue
        src_hart = events[relation.src].hart if relation.src in events else 0
        dst_hart = events[relation.dst].hart if relation.dst in events else 0
        items.append(
            {
                "relation": relation,
                "kind": relation.kind,
                "label": relation.label,
                "src": src,
                "dst": dst,
                "src_hart": src_hart,
                "dst_hart": dst_hart,
                "span": abs(dst_hart - src_hart),
                "ltr": src_hart <= dst_hart,
            }
        )

    # Widest arch sits on the highest lane.
    distant = sorted([it for it in items if it["span"] >= 2], key=lambda it: -it["span"])
    for lane_index, it in enumerate(distant):
        it["lane"] = lane_index
        it["lane_count"] = len(distant)

    # Group endpoints by the box edge they touch so we can spread them out.
    edge_groups: dict[tuple[str, str], list[tuple[dict[str, Any], str]]] = {}
    for it in items:
        src_side, dst_side = ("R", "L") if it["ltr"] else ("L", "R")
        edge_groups.setdefault((it["relation"].src, src_side), []).append((it, "src"))
        edge_groups.setdefault((it["relation"].dst, dst_side), []).append((it, "dst"))

    attach: dict[tuple[int, str], float] = {}
    for (event_id, _side), endpoints in edge_groups.items():
        box = boxes[event_id]

        def sort_key(ep: tuple[dict[str, Any], str]) -> float:
            it, role = ep
            if it["span"] >= 2:
                return -10000 + it["lane"]  # arches leave near the top of the edge
            other = it["dst"] if role == "src" else it["src"]
            return (other[1] + other[3]) / 2

        ordered = sorted(endpoints, key=sort_key)
        n = len(ordered)
        for i, (it, role) in enumerate(ordered):
            frac = 0.5 if n == 1 else 0.30 + 0.40 * (i / (n - 1))
            attach[(id(it["relation"]), role)] = box[1] + frac * (box[3] - box[1])

    gap_seen: dict[int, int] = {}
    routed: list[dict[str, Any]] = []
    for it in items:
        src_y = attach[(id(it["relation"]), "src")]
        dst_y = attach[(id(it["relation"]), "dst")]
        if it["span"] >= 2:
            points = _arch_points(it, columns, src_y, dst_y, it["lane"], it["lane_count"])
            slot = it["lane"]
        else:
            gap_key = min(it["src_hart"], it["dst_hart"])
            slot = gap_seen.get(gap_key, 0)
            gap_seen[gap_key] = slot + 1
            points = _side_curve_points(it, columns, src_y, dst_y, slot)
        routed.append({"points": points, "kind": it["kind"], "label": it["label"], "slot": slot})
    return routed


def _relation_style(kind: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    color = REL_STYLE.get(kind, REL_STYLE["obs"])
    return color, color


def _side_curve_points(
    item: dict[str, Any],
    columns: list[tuple[int, int]],
    src_y: float,
    dst_y: float,
    slot: int,
) -> list[tuple[float, float]]:
    src, dst = item["src"], item["dst"]
    if item["ltr"]:
        start = (src[2], src_y)
        end = (dst[0], dst_y)
    else:
        start = (src[0], src_y)
        end = (dst[2], dst_y)
    gap_mid = (start[0] + end[0]) / 2
    # Fan successive relations in the same gap apart with a small bow offset.
    bow = (slot + 1) * 16 * (1 if slot % 2 == 0 else -1)
    c1 = (gap_mid + bow, src_y)
    c2 = (gap_mid + bow, dst_y)
    return _cubic_points(start, c1, c2, end, steps=40)


def _arch_points(
    item: dict[str, Any],
    columns: list[tuple[int, int]],
    src_y: float,
    dst_y: float,
    lane_index: int,
    lane_count: int,
) -> list[tuple[float, float]]:
    src, dst = item["src"], item["dst"]
    lane_y = max(LANE_TOP, LANE_BASE - lane_index * LANE_STEP)
    # Each arch shifts its vertical channels sideways so arches sharing a gap
    # don't draw on top of one another (centered around the gap midpoint).
    channel = (lane_index - (lane_count - 1) / 2) * 12
    sh, dh = item["src_hart"], item["dst_hart"]
    # Exit toward the destination side; climb in the adjacent gap; cross; descend.
    if item["ltr"]:
        start = (src[2], src_y)
        up_x = _gap_center(columns, sh, sh + 1) + channel
        down_x = _gap_center(columns, dh - 1, dh) + channel
        end = (dst[0], dst_y)
    else:
        start = (src[0], src_y)
        up_x = _gap_center(columns, sh - 1, sh) + channel
        down_x = _gap_center(columns, dh, dh + 1) + channel
        end = (dst[2], dst_y)
    raw = [start, (up_x, src_y), (up_x, lane_y), (down_x, lane_y), (down_x, dst_y), end]
    return _smooth(raw, iterations=3)


def _gap_center(columns: list[tuple[int, int]], left_col: int, right_col: int) -> float:
    left_col = max(0, min(left_col, len(columns) - 1))
    right_col = max(0, min(right_col, len(columns) - 1))
    if left_col == right_col:
        x0, x1 = columns[left_col]
        return (x0 + x1) / 2
    return (columns[left_col][1] + columns[right_col][0]) / 2


def _draw_cycle(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.FreeTypeFont], case_ir: LitmusCaseIR) -> None:
    _round(draw, (70, 766, 1530, 880), COLORS["rel_bg"], COLORS["border"], 12)
    draw.text((95, 784), "Dependency cycle", fill=COLORS["text"], font=fonts["h2"])
    tokens = [relation.label for relation in case_ir.relations]
    if not tokens:
        tokens = [case_ir.cycle]
    x = 95
    y = 828
    for index, token in enumerate(tokens):
        w = min(190, max(80, _text_size(draw, token, fonts["small_b"])[0] + 26))
        if x + w > 1450:
            x = 95
            y += 38
        _round(draw, (x, y, x + w, y + 28), COLORS["panel"], COLORS["border"], 8)
        draw.text((x + 12, y + 5), token, fill=COLORS["text"], font=fonts["small_b"])
        if index < len(tokens) - 1:
            _arrow(draw, (x + w + 4, y + 14), (x + w + 32, y + 14), COLORS["local"])
        x += w + 42


def _draw_footer(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.FreeTypeFont], case_ir: LitmusCaseIR, solver: dict[str, Any] | None) -> None:
    _round(draw, (70, 900, 1530, 1060), COLORS["panel"], COLORS["border"], 12)
    draw.text((95, 920), "Outcome", fill=COLORS["text"], font=fonts["h2"])
    verdict = (solver or {}).get("verdict", "unchecked")
    status = (solver or {}).get("status", "not_applicable")
    reason = (solver or {}).get("reason", "No solver result attached.")
    lines = [f"exists {case_ir.exists}", f"solver: {_solver_badge(solver)}; raw status: {status}; verdict: {verdict}", str(reason)]
    y = 955
    for line in lines:
        for wrapped in _wrap(draw, line, fonts["body"], 1340):
            draw.text((95, y), wrapped, fill=COLORS["text"], font=fonts["body"])
            y += 27


def _event_color(kind: str) -> tuple[int, int, int]:
    return {
        "load": COLORS["load"],
        "store": COLORS["store"],
        "vector": COLORS["vector"],
        "cmo": COLORS["cmo"],
        "fence": COLORS["fence"],
        "dep": COLORS["dep"],
        "setup": COLORS["fence"],
    }.get(kind, COLORS["panel"])


def _verdict_color(verdict: str) -> tuple[int, int, int]:
    if verdict == "forbidden":
        return (254, 226, 226)
    if verdict == "allowed":
        return (220, 252, 231)
    return (241, 245, 249)


def _solver_badge(solver: dict[str, Any] | None) -> str:
    status = (solver or {}).get("status", "not_applicable")
    verdict = (solver or {}).get("verdict", "unchecked")
    cross = (solver or {}).get("cross_check", "")
    if status == "verified":
        if cross == "agree":
            return f"{verdict} (native+herd7)"
        return f"{verdict} (native)"
    if status == "conflict":
        return f"conflict: {verdict}"
    if status == "not_applicable":
        return "observation only"
    return f"{status}: {verdict}"


def _cubic_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int = 40,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def _smooth(points: list[tuple[float, float]], iterations: int = 2) -> list[tuple[float, float]]:
    """Chaikin corner-cutting; keeps the endpoints so arrowheads stay aligned."""
    result = points
    for _ in range(iterations):
        if len(result) < 3:
            break
        smoothed = [result[0]]
        for a, b in zip(result, result[1:]):
            smoothed.append((a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25))
            smoothed.append((a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75))
        smoothed.append(result[-1])
        result = smoothed
    return result


def _path_point_at_fraction(points: list[tuple[float, float]], fraction: float) -> tuple[float, float]:
    if len(points) < 2:
        return points[0]
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    total = sum(lengths) or 1.0
    target = total * max(0.0, min(1.0, fraction))
    acc = 0.0
    for (a, b), seg in zip(zip(points, points[1:]), lengths):
        if acc + seg >= target:
            t = (target - acc) / seg if seg else 0.0
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        acc += seg
    return points[-1]


def _draw_curve(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.FreeTypeFont],
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    accent: tuple[int, int, int],
    label: str,
    slot: int,
) -> None:
    draw.line(points, fill=color, width=4, joint="curve")
    _arrowhead(draw, points[-2], points[-1], color, size=14)
    chip_fraction = [0.5, 0.4, 0.6, 0.34, 0.66][slot % 5]
    cx, cy = _path_point_at_fraction(points, chip_fraction)
    _chip(draw, fonts, (cx, cy), label, accent)


def _chip(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.FreeTypeFont],
    center: tuple[float, float],
    label: str,
    accent: tuple[int, int, int],
) -> None:
    font = fonts["small_b"]
    tw, th = _text_size(draw, label, font)
    pad_x, pad_y = 9, 5
    cx, cy = center
    box = (cx - tw / 2 - pad_x, cy - th / 2 - pad_y, cx + tw / 2 + pad_x, cy + th / 2 + pad_y)
    draw.rounded_rectangle(box, radius=7, fill=COLORS["panel"], outline=accent, width=2)
    draw.text((cx - tw / 2, cy - th / 2 - 1), label, fill=accent, font=font)


def _round(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int], outline: tuple[int, int, int], radius: int) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], label: str = "", font: ImageFont.FreeTypeFont | None = None) -> None:
    draw.line([start, end], fill=color, width=3)
    _arrowhead(draw, start, end, color)
    if label and font is not None:
        draw.text(((start[0] + end[0]) // 2 + 6, (start[1] + end[1]) // 2 - 12), label, fill=color, font=font)


def _arrowhead(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int], size: int = 13) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (end[0] - size * math.cos(angle - math.pi / 6), end[1] - size * math.sin(angle - math.pi / 6))
    right = (end[0] - size * math.cos(angle + math.pi / 6), end[1] - size * math.sin(angle + math.pi / 6))
    draw.polygon([end, left, right], fill=color)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], font: ImageFont.FreeTypeFont, fill: tuple[int, int, int]) -> None:
    lines = _wrap(draw, text, font, box[2] - box[0])
    y = box[1]
    for line in lines[:2]:
        draw.text((box[0], y), line, fill=fill, font=font)
        y += 18


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]
