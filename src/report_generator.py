"""
src/report_generator.py
Clinical-grade PDF Report Generator for Biometric Blood Group Identification.
Uses ReportLab to generate standard medical estimation reports.
"""

import os
import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable
)

from src.config import PROJECT_ROOT, ADVISORY_CONFIRMATION_THRESHOLD

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

MANDATORY_DISCLAIMER = (
    "RESEARCH / PROTOTYPE SYSTEM DISCLAIMER: This system is a research/prototype fingerprint-based blood-group classification system. "
    "It is not clinically validated and must not be used as a substitute for standard serological blood-group testing or for medical/transfusion decisions. "
    "All predictions and advisory indicators are provided solely for evaluation purposes."
)

def generate_pdf_report(
    prediction_id,
    timestamp,
    input_type,
    image_path,
    predicted_group,
    confidence,
    probabilities=None,
    status="NORMAL",
    model_name=None,
    notes="",
    output_filename=None,
    finger_used=None,
    advisory_level=None,
    advisory_reason=None,
    advisory_message=None,
    advisory_threshold=ADVISORY_CONFIRMATION_THRESHOLD,
    needs_confirmation=False,
    crop_metadata=None
):
    """
    Generates a formal PDF report.
    Returns:
        output_path: Absolute path to the generated PDF.
    """
    if output_filename is None:
        clean_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"blood_group_report_{prediction_id}_{clean_ts}.pdf"

    output_path = os.path.join(REPORTS_DIR, output_filename)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e3a8a"),
        alignment=1 # Center
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#4b5563"),
        alignment=1
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1f2937")
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#991b1b")
    )
    table_text = ParagraphStyle(
        "TableText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#111827")
    )
    table_bold = ParagraphStyle(
        "TableBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#111827")
    )
    result_blood_group = ParagraphStyle(
        "ResultBloodGroup",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=32,
        leading=38,
        textColor=colors.HexColor("#dc2626"),
        alignment=1
    )
    result_meta = ParagraphStyle(
        "ResultMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#374151"),
        alignment=1
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("BIOMETRIC BLOOD GROUP IDENTIFICATION SYSTEM", title_style))
    elements.append(Paragraph("Final Year B.E. Major Project • Deep Learning AI Analysis Report", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=12))

    # 2. Medical Disclaimer Box
    disclaimer_table = Table(
        [[Paragraph(MANDATORY_DISCLAIMER, disclaimer_style)]],
        colWidths=[540]
    )
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fee2e2")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ef4444")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    elements.append(disclaimer_table)
    elements.append(Spacer(1, 14))
    # Model name will be shown in the "Model Name" table row below

    # 3. Sample / Examination Details
    elements.append(Paragraph("1. EXAMINATION & SAMPLE AUDIT DETAILS", section_heading))
    elements.append(Spacer(1, 6))

    info_data = [
        [
            Paragraph("<b>Report ID:</b>", table_bold),
            Paragraph(f"BG-{prediction_id:06d}", table_text),
            Paragraph("<b>Timestamp:</b>", table_bold),
            Paragraph(str(timestamp), table_text)
        ],
        [
            Paragraph("<b>Input Source:</b>", table_bold),
            Paragraph(str(input_type), table_text),
            Paragraph("<b>Evaluation Status:</b>", table_bold),
            Paragraph(f"<font color='{'green' if status=='NORMAL' else 'red'}'><b>{status}</b></font>", table_text)
        ],
        [
            Paragraph("<b>Model Name:</b>", table_bold),
            Paragraph(str(model_name), table_text),
            Paragraph("<b>Confidence:</b>", table_bold),
            Paragraph(f"<b>{confidence * 100:.2f}%</b>", table_text)
        ]
    ]

    # Append Standard Acquisition Finger row for physical MFS100 captures
    if finger_used:
        finger_style = ParagraphStyle(
            "FingerUsed",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1d4ed8")
        )
        info_data.append([
            Paragraph("<b>Standard Acquisition Finger:</b>", table_bold),
            Paragraph(
                f"<font color='#1d4ed8'><b>{finger_used.replace('_', ' ')}</b></font> "
                "<font color='#6b7280'>(Convention-based protocol — not automatically verified by scanner)</font>",
                table_text
            ),
            Paragraph("", table_text),
            Paragraph("", table_text)
        ])

    # Advisory screening status
    eff_adv_level = advisory_level if advisory_level else ("NEEDS_CONFIRMATION" if (needs_confirmation or confidence < advisory_threshold) else "CONFIDENCE_OK")
    adv_badge_text = "CONFIDENCE OK" if eff_adv_level == "CONFIDENCE_OK" else "NEEDS CONFIRMATION"
    adv_color = "#15803d" if eff_adv_level == "CONFIDENCE_OK" else "#b45309"
    adv_reason_str = advisory_reason if advisory_reason else ("NONE" if eff_adv_level == "CONFIDENCE_OK" else "LOW_CONFIDENCE")

    info_data.append([
        Paragraph("<b>Advisory Status:</b>", table_bold),
        Paragraph(f"<font color='{adv_color}'><b>{adv_badge_text}</b></font> (&tau;={advisory_threshold*100:.0f}%)", table_text),
        Paragraph("<b>Advisory Reason:</b>", table_bold),
        Paragraph(f"<b>{adv_reason_str}</b>", table_text)
    ])

    t_style = [
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f9fafb")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]
    if finger_used:
        t_style.append(("SPAN", (1, len(info_data) - 2), (3, len(info_data) - 2)))

    info_table = Table(info_data, colWidths=[110, 160, 110, 160])
    info_table.setStyle(TableStyle(t_style))
    elements.append(info_table)
    elements.append(Spacer(1, 16))

    # 4. Fingerprint Impression and Classification Card
    elements.append(Paragraph("2. BIOMETRIC IMPRESSION & PREDICTED BLOOD GROUP", section_heading))
    elements.append(Spacer(1, 8))

    # Image element
    img_flowable = None
    if image_path and os.path.exists(image_path):
        try:
            # Convert to temporary RGB PNG for ReportLab if needed
            temp_png = os.path.join(REPORTS_DIR, f"temp_preview_{prediction_id}.png")
            with Image.open(image_path) as im:
                im.convert("RGB").save(temp_png, "PNG")
            img_flowable = RLImage(temp_png, width=140, height=140)
        except Exception:
            img_flowable = Paragraph("<i>[Image preview unavailable]</i>", table_text)
    else:
        img_flowable = Paragraph("<i>[Image not available]</i>", table_text)

    # Detailed ABO decomposition summary text
    meta_lines = [
        f"Confidence Score: <b>{confidence * 100:.2f}%</b>",
        f"Advisory Threshold: <b>{advisory_threshold * 100:.0f}%</b>",
        f"Advisory Screening: <font color='{adv_color}'><b>{adv_badge_text}</b></font>"
    ]
    if crop_metadata and crop_metadata.get("crop_applied"):
        bbox = crop_metadata.get("bbox", [0, 0, 0, 0])
        orig = crop_metadata.get("orig_size", [0, 0])
        meta_lines.append(f"Original: {orig[0]}x{orig[1]}px &rarr; BBox: ({bbox[0]}, {bbox[1]}) &rarr; ({bbox[2]}, {bbox[3]})")

    result_card_data = [
        Paragraph(f"Predicted Blood Group:<br/><br/><b>{predicted_group}</b>", result_blood_group),
        Paragraph("<br/>".join(meta_lines), result_meta)
    ]

    card_data = [
        [img_flowable, result_card_data]
    ]

    card_table = Table(card_data, colWidths=[170, 370])
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f3f4f6")),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#eff6ff")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 10)
    ]))
    elements.append(card_table)

    # Advisory Guidance Banner
    elements.append(Spacer(1, 10))
    if eff_adv_level == "NEEDS_CONFIRMATION":
        adv_banner_msg = advisory_message if advisory_message else (
            "Prediction confidence is below the validation-selected advisory threshold of 70%. "
            "Consider repeated fingerprint capture or 3-scan consensus. "
            "Verify blood group using standard serological testing before any medical or transfusion-related use."
        )
        adv_table = Table(
            [[Paragraph(f"<b>RESEARCH ADVISORY &mdash; CONFIRMATION RECOMMENDED:</b> {adv_banner_msg}", ParagraphStyle("AdvWarn", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#78350f")))]],
            colWidths=[540]
        )
        adv_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#f59e0b")),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))
        elements.append(adv_table)
    else:
        adv_banner_msg = "Advisory threshold passed (&ge; 70.0%). Note: Confidence OK does not constitute medical verification. Standard serological testing required for medical/transfusion use."
        adv_table = Table(
            [[Paragraph(f"<b>RESEARCH ADVISORY:</b> {adv_banner_msg}", ParagraphStyle("AdvOk", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#166534")))]],
            colWidths=[540]
        )
        adv_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#dcfce7")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#22c55e")),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))
        elements.append(adv_table)




    elements.append(Spacer(1, 16))

    # 5. Full Probability Distribution Table
    elements.append(Paragraph("3. 4-CLASS BLOOD GROUP PROBABILITY DISTRIBUTION", section_heading))
    elements.append(Spacer(1, 8))

    if probabilities and isinstance(probabilities, dict):
        prob_headers = [Paragraph(f"<b>{k}</b>", table_bold) for k in probabilities.keys()]
        prob_values = [Paragraph(f"{v * 100:.1f}%", table_text) for v in probabilities.values()]
        prob_data = [prob_headers, prob_values]

        prob_table = Table(prob_data, colWidths=[135] * 4)
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#ffffff")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 5)
        ]))
        elements.append(prob_table)
    else:
        elements.append(Paragraph("<i>Probability distribution not available</i>", table_text))

    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#9ca3af"), spaceAfter=8))
    
    # Footer
    elements.append(Paragraph(
        "Biometric Blood Group Identification System • Research & Academic Demonstration Platform • Mantra MFS100 Optical Sensor Compatible",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#6b7280"), alignment=1)
    ))

    # Build Document
    doc.build(elements)

    

    # Clean up temp image
    temp_png = os.path.join(REPORTS_DIR, f"temp_preview_{prediction_id}.png")
    if os.path.exists(temp_png):
        try:
            os.remove(temp_png)
        except Exception:
            pass

    return output_path
