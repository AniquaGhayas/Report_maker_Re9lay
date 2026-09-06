"""
Re9lay - Assembles the final PDF progress report using ReportLab,
embedding the matplotlib charts and a grouped metrics summary.
Includes a branded header (with optional logo), page border, and
page numbers via a canvas page-decoration callback.
"""

import os
from datetime import datetime
from functools import partial

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
    KeepTogether,
)

# --- Color palette ---
NAVY = colors.HexColor("#0f2d5c")
BLUE = colors.HexColor("#2a6fdb")
LIGHT_BLUE = colors.HexColor("#eaf1fb")
CARD_BLUE = colors.HexColor("#f2f6fc")
GREY_TEXT = colors.HexColor("#5a6b7d")
BORDER_GREY = colors.HexColor("#c9d6e8")
ACCENT = {
    "score": colors.HexColor("#2a6fdb"),
    "emg": colors.HexColor("#c0562b"),
    "motion": colors.HexColor("#6a3fb3"),
}

PAGE_W, PAGE_H = letter
MARGIN = 0.55 * inch
HEADER_H = 0.9 * inch

PRODUCT_NAME = "Re9lay"


def _fmt(val, unit=""):
    if isinstance(val, float):
        return f"{val:.2f}{unit}"
    return f"{val}{unit}"


def _logo_box_size(logo_path, max_w, max_h):
    """Return (w, h) that fits the logo's real aspect ratio inside the box."""
    try:
        from PIL import Image as PILImage
        with PILImage.open(logo_path) as im:
            iw, ih = im.size
        ratio = iw / ih
        if max_w / max_h > ratio:
            return max_h * ratio, max_h
        return max_w, max_w / ratio
    except Exception:
        return max_h, max_h


def _page_decoration(canvas, doc, session_label, logo_path):
    """Draws the border, header band (with logo + title), and footer
    page number on every page."""
    canvas.saveState()

    # --- outer page border ---
    canvas.setStrokeColor(BLUE)
    canvas.setLineWidth(1.2)
    canvas.rect(0.3 * inch, 0.3 * inch, PAGE_W - 0.6 * inch, PAGE_H - 0.6 * inch)

    # --- header band (white, with a bottom rule) ---
    header_bottom = PAGE_H - 0.3 * inch - HEADER_H
    canvas.setFillColor(colors.white)
    canvas.rect(0.3 * inch, header_bottom, PAGE_W - 0.6 * inch, HEADER_H, fill=1, stroke=0)
    canvas.setStrokeColor(BLUE)
    canvas.setLineWidth(1.4)
    canvas.line(0.3 * inch, header_bottom, PAGE_W - 0.3 * inch, header_bottom)

    text_x = MARGIN + 0.15 * inch
    if logo_path and os.path.exists(logo_path):
        max_logo_w, max_logo_h = 0.7 * inch, 0.6 * inch
        lw, lh = _logo_box_size(logo_path, max_logo_w, max_logo_h)
        canvas.drawImage(
            logo_path,
            MARGIN + 0.1 * inch,
            header_bottom + HEADER_H / 2 - lh / 2,
            width=lw, height=lh,
            preserveAspectRatio=True, mask="auto",
        )
        text_x = MARGIN + 0.1 * inch + max_logo_w + 0.2 * inch

    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 17)
    canvas.drawString(text_x, header_bottom + HEADER_H / 2 + 4, PRODUCT_NAME)
    canvas.setFont("Helvetica", 10.5)
    canvas.setFillColor(GREY_TEXT)
    canvas.drawString(text_x, header_bottom + HEADER_H / 2 - 12, "Progress Report")

    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(GREY_TEXT)
    canvas.drawRightString(PAGE_W - MARGIN - 0.15 * inch,
                            header_bottom + HEADER_H / 2, session_label)

    # --- footer: page number ---
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(GREY_TEXT)
    canvas.drawCentredString(PAGE_W / 2, 0.42 * inch, f"Page {doc.page}")
    canvas.drawString(MARGIN, 0.42 * inch, PRODUCT_NAME)

    canvas.restoreState()


def _stat_card(label, value, styles, accent, width=2.15):
    """One metric as a bordered card with a colored top accent bar."""
    label_style = ParagraphStyle(
        "CardLabel", parent=styles["Normal"], fontSize=9,
        textColor=GREY_TEXT, leading=12,
    )
    value_style = ParagraphStyle(
        "CardValue", parent=styles["Normal"], fontSize=15.5,
        textColor=NAVY, fontName="Helvetica-Bold", leading=19, spaceBefore=4,
    )
    cell = Table(
        [[Paragraph(label, label_style)], [Paragraph(value, value_style)]],
        colWidths=[width * inch],
    )
    cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.75, BORDER_GREY),
        ("LINEABOVE", (0, 0), (-1, 0), 2.5, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    return cell


def _stat_group(title, stats, accent, styles, doc_width, per_row=None):
    """A labeled section: subheading + a row-wrapped grid of stat cards.
    Groups of 4 or fewer render as a single tidy row; larger groups wrap
    at 3 per row so no row is left with a single orphaned card."""
    if per_row is None:
        per_row = len(stats) if len(stats) <= 5 else 3
    heading_style = ParagraphStyle(
        "GroupHeading", parent=styles["Heading3"], textColor=accent,
        fontSize=12.5, spaceBefore=0, spaceAfter=10,
    )
    flow = [Paragraph(title, heading_style)]
    card_w = (doc_width / inch) / per_row - 0.12
    cards = [_stat_card(label, value, styles, accent, width=card_w) for label, value in stats]
    rows = [cards[i:i + per_row] for i in range(0, len(cards), per_row)]
    grid = Table(rows, colWidths=[card_w * inch] * per_row, hAlign="LEFT")
    grid.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    flow.append(grid)
    return [KeepTogether(flow), Spacer(1, 20)]


def build_report(metrics: dict, chart_paths: dict, out_path: str,
                  session_label: str = None, logo_path: str = None):
    styles = getSampleStyleSheet()
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], textColor=NAVY,
        fontSize=16, spaceBefore=4, spaceAfter=14,
    )
    chart_heading_style = ParagraphStyle(
        "ChartHeading", parent=styles["Heading2"], textColor=NAVY,
        fontSize=13, spaceBefore=2, spaceAfter=6,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=8.5, textColor=GREY_TEXT,
        alignment=TA_CENTER, spaceAfter=4, spaceBefore=4,
    )

    doc = SimpleDocTemplate(
        out_path, pagesize=letter,
        topMargin=0.3 * inch + HEADER_H + 0.3 * inch,
        bottomMargin=0.65 * inch,
        leftMargin=MARGIN, rightMargin=MARGIN,
    )
    story = []

    session_label = session_label or datetime.now().strftime("%Y-%m-%d %H:%M")

    def section_title(text):
        t = Table([[Paragraph(text, section_style)]], colWidths=[doc.width])
        t.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 1.5, BLUE),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return t

    story.append(section_title("Session Summary"))
    story.append(Spacer(1, 14))

    # --- Summary grouped into three labeled sections ---
    emg, motion, score, acc = metrics["emg"], metrics["motion"], metrics["score"], metrics["accuracy"]

    story += _stat_group(
        "Score & Performance",
        [
            ("Session Duration", _fmt(score["session_duration_s"], " s")),
            ("Final Score", _fmt(score["final_score"])),
            ("Score Rate", _fmt(score["score_rate_per_sec"], " pts/s")),
            ("Shot Hit Rate", _fmt(acc["hit_rate_pct"], " %")),
        ],
        ACCENT["score"], styles, doc.width,
    )

    story += _stat_group(
        "Muscle Activity (EMG)",
        [
            ("Contraction Count", _fmt(emg["contraction_count"])),
            ("Mean Contraction Duration", _fmt(emg["mean_contraction_duration_s"], " s")),
            ("EMG Duty Cycle", _fmt(emg["duty_cycle_pct"], " %")),
            ("Contraction Frequency", _fmt(emg["contraction_freq_per_min"], " /min")),
            ("Mean EMG", _fmt(emg["mean_emg"])),
            ("Peak EMG", _fmt(emg["peak_emg"])),
        ],
        ACCENT["emg"], styles, doc.width,
    )

    story += _stat_group(
        "Movement & Motion",
        [
            ("Total Distance Traveled", _fmt(motion["total_distance"], " units")),
            ("Mean Speed", _fmt(motion["mean_speed"])),
            ("Peak Speed", _fmt(motion["peak_speed"])),
            ("Path Efficiency", _fmt(motion["path_efficiency_pct"], " %")),
            ("Movement Smoothness (LDLJ)", _fmt(motion["overall_ldlj"])),
        ],
        ACCENT["motion"], styles, doc.width,
    )

    story.append(PageBreak())

    # --- Charts, two per page with captions ---
    chart_sections = [
        ("trajectory", "Movement Trajectory & Direction",
         "Arrows show movement direction; color indicates instantaneous speed."),
        ("range_of_motion", "Range of Motion",
         "Maximum pitch/roll deviation in each direction during the session."),
        ("score_vs_time", "Score Progression",
         "Dashed lines mark each game-speed tier transition."),
        ("emg_threshold", "EMG Signal & Contraction Detection",
         "Shaded regions indicate detected muscle contractions above threshold."),
        ("kinematics", "Movement Kinematics",
         "Speed and acceleration over the session, with LDLJ (log dimensionless jerk) "
         "as the movement smoothness index - values closer to 0 indicate smoother movement."),
    ]

    pairs = [chart_sections[i:i + 2] for i in range(0, len(chart_sections), 2)]
    for p_idx, pair in enumerate(pairs):
        for key, heading, caption in pair:
            story.append(Paragraph(heading, chart_heading_style))
            story.append(Image(chart_paths[key], width=5.2 * inch, height=3.15 * inch,
                                kind="proportional"))
            story.append(Paragraph(caption, caption_style))
            story.append(Spacer(1, 10))
        if p_idx < len(pairs) - 1:
            story.append(PageBreak())

    decorate = partial(_page_decoration, session_label=session_label, logo_path=logo_path)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return out_path
