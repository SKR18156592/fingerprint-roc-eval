"""Render docs/report.md to docs/report.pdf.

Supports the small Markdown subset the report uses: #/##/### headings,
paragraphs, "- " bullets, "1. " numbered items, pipe tables, fenced code
blocks, **bold**, `code`, and ![caption](path) images. Keeping the report in
Markdown means it also renders directly on GitHub.

Usage:
    python scripts/build_report.py [docs/report.md] [docs/report.pdf]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=styles["BodyText"], fontSize=8.8, leading=11.4, spaceAfter=3)
H1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceAfter=4, spaceBefore=0)
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=7, spaceAfter=3)
H3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=10, spaceBefore=5, spaceAfter=2)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8, alignment=TA_CENTER, textColor=colors.grey)
CODE = ParagraphStyle("code", parent=styles["Code"], fontSize=7.8, leading=9.5)


def inline(text: str) -> str:
    """Escape XML, then convert **bold**, *italic* and `code`."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def table(lines: list[str], width: float) -> Table:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    data = [[Paragraph(inline(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[width / len(rows[0])] * len(rows[0]), repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b8c4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


MAX_IMAGE_HEIGHT = 5.4 * cm  # keeps figures from eating a whole page


def image_row(images: list[tuple[str, str]], base: Path, width: float) -> Table:
    cells, caps = [], []
    w = width / len(images) - 0.2 * cm
    for caption, src in images:
        img = Image(str((base / src).resolve()))
        scale = min(w / img.drawWidth, MAX_IMAGE_HEIGHT / img.drawHeight)
        img.drawWidth, img.drawHeight = img.drawWidth * scale, img.drawHeight * scale
        cells.append(img)
        caps.append(Paragraph(inline(caption), CAPTION))
    return Table([cells, caps], colWidths=[width / len(images)] * len(images))


def build(md_path: Path, pdf_path: Path) -> None:
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm)
    width = doc.width
    lines = md_path.read_text().splitlines()
    story, para, i = [], [], 0

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), BODY))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            story.append(Preformatted("\n".join(lines[i + 1:j]), CODE))
            story.append(Spacer(1, 4))
            i = j + 1
            continue
        if not s or s.startswith("<!--"):
            flush()
        elif s.startswith("# "):
            flush(); story.append(Paragraph(inline(s[2:]), H1))
        elif s.startswith("## "):
            flush(); story.append(Paragraph(inline(s[3:]), H2))
        elif s.startswith("### "):
            flush(); story.append(Paragraph(inline(s[4:]), H3))
        elif s.startswith("|"):
            flush()
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            story.append(table(lines[i:j], width))
            story.append(Spacer(1, 5))
            i = j
            continue
        elif s.startswith("!["):
            flush()
            imgs = re.findall(r"!\[(.*?)\]\((.*?)\)", s)
            story.append(image_row(imgs, md_path.parent, width))
            story.append(Spacer(1, 4))
        elif re.match(r"^(- |\d+\. )", s):
            flush()
            numbered = bool(re.match(r"^\d+\. ", s))
            items = []
            while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i].strip()):
                text = re.sub(r"^(- |\d+\. )", "", lines[i].strip())
                # continuation lines indented under the item
                while i + 1 < len(lines) and lines[i + 1].startswith("   ") and lines[i + 1].strip():
                    i += 1
                    text += " " + lines[i].strip()
                items.append(ListItem(Paragraph(inline(text), BODY), leftIndent=12))
                i += 1
            story.append(ListFlowable(items, bulletType="1" if numbered else "bullet",
                                      leftIndent=12, bulletFontSize=8))
            continue
        else:
            para.append(s)
        i += 1
    flush()
    doc.build(story)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    md = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/report.md")
    pdf = Path(sys.argv[2] if len(sys.argv) > 2 else md.with_suffix(".pdf"))
    build(md, pdf)
