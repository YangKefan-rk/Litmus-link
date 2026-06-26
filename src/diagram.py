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
}


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
    event_boxes = _draw_harts(draw, fonts, case_ir)
    _draw_relations(draw, fonts, case_ir, event_boxes)
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
    status = (solver or {}).get("status", "not_applicable")
    verdict = (solver or {}).get("verdict", "unchecked")
    badge = f"{status} / {verdict}"
    _round(draw, (1230, 55, 1515, 100), _verdict_color(verdict), COLORS["border"], 10)
    draw.text((1252, 66), badge, fill=COLORS["text"], font=fonts["h2"])


def _draw_harts(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.FreeTypeFont], case_ir: LitmusCaseIR) -> dict[str, tuple[int, int, int, int]]:
    hart_count = max(len(case_ir.harts), 1)
    left, top, right, bottom = 70, 180, 1530, 700
    gap = 28
    col_width = int((right - left - gap * (hart_count - 1)) / hart_count)
    event_boxes: dict[str, tuple[int, int, int, int]] = {}
    for hart_index, events in enumerate(case_ir.harts):
        x0 = left + hart_index * (col_width + gap)
        x1 = x0 + col_width
        _round(draw, (x0, top, x1, bottom), COLORS["panel"], COLORS["border"], 12)
        draw.text((x0 + 18, top + 15), f"P{hart_index}", fill=COLORS["blue"], font=fonts["h1"])
        y = top + 62
        previous_center = None
        for event in events:
            box_h = 64
            box = (x0 + 18, y, x1 - 18, y + box_h)
            _round(draw, box, _event_color(event.kind), COLORS["border"], 8)
            label = f"{event.event_id}: {event.instruction}"
            _fit_text(draw, label, (box[0] + 12, box[1] + 9, box[2] - 12, box[3] - 28), fonts["mono"], COLORS["text"])
            detail = "  ".join(part for part in [event.kind, event.location, event.role] if part)
            draw.text((box[0] + 12, box[3] - 23), detail, fill=COLORS["muted"], font=fonts["small"])
            center = ((box[0] + box[2]) // 2, box[3])
            if previous_center is not None:
                _arrow(draw, previous_center, ((box[0] + box[2]) // 2, box[1]), COLORS["local"], label="local", font=fonts["small"])
            previous_center = center
            event_boxes[event.event_id] = box
            y += box_h + 36
    return event_boxes


def _draw_relations(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.FreeTypeFont], case_ir: LitmusCaseIR, boxes: dict[str, tuple[int, int, int, int]]) -> None:
    for index, relation in enumerate(case_ir.relations):
        if relation.local:
            continue
        src = boxes.get(relation.src)
        dst = boxes.get(relation.dst)
        if src is None or dst is None:
            continue
        start = (src[2], (src[1] + src[3]) // 2)
        end = (dst[0], (dst[1] + dst[3]) // 2)
        if start[0] > end[0]:
            start = (src[0], (src[1] + src[3]) // 2)
            end = (dst[2], (dst[1] + dst[3]) // 2)
        mid_y = min(start[1], end[1]) - 34 - index * 12
        points = [start, ((start[0] + end[0]) // 2, mid_y), end]
        _polyline_arrow(draw, points, COLORS["rel"], relation.label, fonts["small_b"])


def _draw_cycle(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.FreeTypeFont], case_ir: LitmusCaseIR) -> None:
    _round(draw, (70, 735, 1530, 855), COLORS["rel_bg"], COLORS["border"], 12)
    draw.text((95, 755), "Dependency cycle", fill=COLORS["text"], font=fonts["h2"])
    tokens = [relation.label for relation in case_ir.relations]
    if not tokens:
        tokens = [case_ir.cycle]
    x = 95
    y = 800
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
    _round(draw, (70, 885, 1530, 1040), COLORS["panel"], COLORS["border"], 12)
    draw.text((95, 905), "Outcome", fill=COLORS["text"], font=fonts["h2"])
    verdict = (solver or {}).get("verdict", "unchecked")
    status = (solver or {}).get("status", "not_applicable")
    reason = (solver or {}).get("reason", "No solver result attached.")
    lines = [f"exists {case_ir.exists}", f"solver: {status}; verdict: {verdict}", str(reason)]
    y = 938
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


def _round(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int], outline: tuple[int, int, int], radius: int) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], label: str = "", font: ImageFont.FreeTypeFont | None = None) -> None:
    draw.line([start, end], fill=color, width=3)
    _arrowhead(draw, start, end, color)
    if label and font is not None:
        draw.text(((start[0] + end[0]) // 2 + 6, (start[1] + end[1]) // 2 - 12), label, fill=color, font=font)


def _polyline_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: tuple[int, int, int], label: str, font: ImageFont.FreeTypeFont) -> None:
    for start, end in zip(points, points[1:]):
        draw.line([start, end], fill=color, width=3)
    _arrowhead(draw, points[-2], points[-1], color)
    label_x = points[len(points) // 2][0] + 8
    label_y = points[len(points) // 2][1] - 20
    _round(draw, (label_x - 4, label_y - 2, label_x + _text_size(draw, label, font)[0] + 12, label_y + 22), COLORS["panel"], COLORS["rel"], 6)
    draw.text((label_x + 2, label_y + 1), label, fill=color, font=font)


def _arrowhead(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 13
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
