# ============================================================
# CODTECH INTERNSHIP - TASK 2
# Automated Report Generation
# Author: Sudeep
# Description: Reads student performance data from a CSV,
#              performs analysis, and generates a formatted
#              multi-page PDF report using ReportLab.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import io
import os
from datetime import datetime

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ─────────────────────────────────────────
# STEP 1: GENERATE SAMPLE DATASET
# ─────────────────────────────────────────
def generate_student_data():
    """Create a realistic student performance dataset."""
    np.random.seed(42)
    n = 60
    names = [f"Student_{i:02d}" for i in range(1, n+1)]
    departments = np.random.choice(["Computer Science", "Electronics", "Mechanical", "Civil"], n)
    semesters = np.random.choice([1, 2, 3, 4, 5, 6], n)

    data = {
        "Student Name": names,
        "Department": departments,
        "Semester": semesters,
        "Mathematics": np.random.randint(45, 100, n),
        "Programming": np.random.randint(50, 100, n),
        "Data Structures": np.random.randint(40, 100, n),
        "DBMS": np.random.randint(45, 100, n),
        "Networks": np.random.randint(40, 100, n),
        "Attendance (%)": np.round(np.random.uniform(60, 100, n), 1),
    }

    df = pd.DataFrame(data)
    df["Total"] = df[["Mathematics", "Programming", "Data Structures", "DBMS", "Networks"]].sum(axis=1)
    df["Average"] = (df["Total"] / 5).round(2)
    df["Grade"] = pd.cut(df["Average"],
                         bins=[0, 40, 50, 60, 70, 80, 100],
                         labels=["F", "D", "C", "B", "A", "A+"])
    df["Status"] = df["Average"].apply(lambda x: "Pass" if x >= 50 else "Fail")
    return df


# ─────────────────────────────────────────
# STEP 2: GENERATE CHART IMAGES IN MEMORY
# ─────────────────────────────────────────
def create_chart(fig):
    """Convert matplotlib figure to bytes buffer for ReportLab."""
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def chart_avg_by_dept(df):
    """Bar chart: Average score by department."""
    dept_avg = df.groupby("Department")["Average"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(dept_avg.index, dept_avg.values,
                  color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"], edgecolor="white")
    for bar, val in zip(bars, dept_avg.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=9)
    ax.set_title("Average Score by Department", fontweight="bold")
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    return create_chart(fig)


def chart_grade_distribution(df):
    """Pie chart: Grade distribution."""
    grade_counts = df["Grade"].value_counts()
    colors_list = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.pie(grade_counts.values, labels=grade_counts.index,
           autopct="%1.1f%%", colors=colors_list, startangle=140,
           textprops={"fontsize": 9})
    ax.set_title("Grade Distribution", fontweight="bold")
    fig.tight_layout()
    return create_chart(fig)


def chart_subject_performance(df):
    """Box plot: Score distribution per subject."""
    subjects = ["Mathematics", "Programming", "Data Structures", "DBMS", "Networks"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot([df[s] for s in subjects], labels=subjects,
                    patch_artist=True, notch=False)
    palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title("Subject-wise Score Distribution", fontweight="bold")
    ax.set_ylabel("Marks")
    ax.grid(axis="y", alpha=0.4)
    plt.xticks(rotation=15)
    fig.tight_layout()
    return create_chart(fig)


def chart_attendance_vs_score(df):
    """Scatter: Attendance vs Average score."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    scatter = ax.scatter(df["Attendance (%)"], df["Average"],
                         c=df["Average"], cmap="RdYlGn", s=50, alpha=0.75, edgecolors="grey")
    plt.colorbar(scatter, ax=ax, label="Average Score")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Average Score")
    ax.set_title("Attendance vs Academic Performance", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return create_chart(fig)


# ─────────────────────────────────────────
# STEP 3: BUILD PDF REPORT
# ─────────────────────────────────────────
def build_pdf(df, filename="task2_student_report.pdf"):
    """Build a professional multi-page PDF report."""
    doc = SimpleDocTemplate(
        filename, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()

    # Custom styles
    style_title = ParagraphStyle("Title", parent=styles["Title"],
                                  fontSize=22, textColor=colors.HexColor("#1a1a2e"),
                                  spaceAfter=6, alignment=TA_CENTER)
    style_h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                               fontSize=13, textColor=colors.HexColor("#16213e"),
                               spaceBefore=12, spaceAfter=4)
    style_body = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=9.5, leading=14, spaceAfter=4)
    style_center = ParagraphStyle("Center", parent=style_body, alignment=TA_CENTER)
    style_footer = ParagraphStyle("Footer", parent=styles["Normal"],
                                   fontSize=8, textColor=colors.grey, alignment=TA_CENTER)

    elements = []

    # ── COVER PAGE ──
    elements.append(Spacer(1, 2*cm))
    elements.append(Paragraph("📊 Student Performance Report", style_title))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#4e79a7")))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
                               style_center))
    elements.append(Paragraph("CodTech Internship — Task 2: Automated Report Generation",
                               style_center))
    elements.append(Spacer(1, 1*cm))

    # Summary stats box
    total = len(df)
    passed = (df["Status"] == "Pass").sum()
    failed = total - passed
    top_dept = df.groupby("Department")["Average"].mean().idxmax()
    summary_data = [
        ["Metric", "Value"],
        ["Total Students", str(total)],
        ["Passed", f"{passed} ({passed/total*100:.1f}%)"],
        ["Failed", f"{failed} ({failed/total*100:.1f}%)"],
        ["Overall Average Score", f"{df['Average'].mean():.2f}"],
        ["Highest Score", f"{df['Average'].max():.2f}"],
        ["Lowest Score", f"{df['Average'].min():.2f}"],
        ["Best Performing Department", top_dept],
    ]
    summary_table = Table(summary_data, colWidths=[8*cm, 8*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4e79a7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    elements.append(PageBreak())

    # ── PAGE 2: CHARTS ──
    elements.append(Paragraph("Visual Analysis", style_h2))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    elements.append(Spacer(1, 0.3*cm))

    # Chart 1 & 2 side by side
    img1 = Image(chart_avg_by_dept(df), width=10*cm, height=5*cm)
    img2 = Image(chart_grade_distribution(df), width=7*cm, height=5*cm)
    chart_row = Table([[img1, img2]], colWidths=[10.5*cm, 7.5*cm])
    chart_row.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    elements.append(chart_row)
    elements.append(Spacer(1, 0.4*cm))

    # Chart 3
    elements.append(Paragraph("Subject-wise Score Distribution", style_h2))
    elements.append(Image(chart_subject_performance(df), width=16*cm, height=6*cm))
    elements.append(Spacer(1, 0.4*cm))

    # Chart 4
    elements.append(Paragraph("Attendance vs Academic Performance", style_h2))
    elements.append(Image(chart_attendance_vs_score(df), width=14*cm, height=5.5*cm))
    elements.append(PageBreak())

    # ── PAGE 3: TOP 10 STUDENTS TABLE ──
    elements.append(Paragraph("Top 10 Performing Students", style_h2))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    elements.append(Spacer(1, 0.3*cm))

    top10 = df.nlargest(10, "Average")[
        ["Student Name", "Department", "Average", "Grade", "Attendance (%)", "Status"]
    ].reset_index(drop=True)

    table_data = [["#", "Name", "Department", "Avg", "Grade", "Attend %", "Status"]]
    for i, row in top10.iterrows():
        table_data.append([
            str(i+1), row["Student Name"], row["Department"],
            f"{row['Average']:.1f}", str(row["Grade"]),
            f"{row['Attendance (%)']:.1f}%", row["Status"]
        ])

    student_table = Table(table_data, colWidths=[1*cm, 3.5*cm, 4*cm, 2*cm, 2*cm, 2.5*cm, 2*cm])
    student_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eaf4fb"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(student_table)
    elements.append(Spacer(1, 0.8*cm))

    # Department-wise table
    elements.append(Paragraph("Department-wise Performance Summary", style_h2))
    dept_summary = df.groupby("Department").agg(
        Students=("Student Name", "count"),
        Avg_Score=("Average", "mean"),
        Pass_Rate=("Status", lambda x: f"{(x=='Pass').mean()*100:.1f}%"),
        Top_Score=("Average", "max")
    ).reset_index()

    dept_data = [["Department", "Students", "Avg Score", "Pass Rate", "Top Score"]]
    for _, row in dept_summary.iterrows():
        dept_data.append([row["Department"], str(row["Students"]),
                          f"{row['Avg_Score']:.2f}", row["Pass_Rate"],
                          f"{row['Top_Score']:.2f}"])

    dept_table = Table(dept_data, colWidths=[5*cm, 3*cm, 3*cm, 3*cm, 3*cm])
    dept_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4e79a7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    elements.append(dept_table)
    elements.append(Spacer(1, 1*cm))

    # Footer note
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    elements.append(Spacer(1, 0.2*cm))
    elements.append(Paragraph(
        "This report was auto-generated by the CodTech Python Internship Task 2 script. "
        "Data analysis performed using Pandas and visualizations using Matplotlib.",
        style_footer
    ))

    doc.build(elements)
    print(f"✅ PDF report saved as '{filename}'")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("📊 Generating student performance data...")
    df = generate_student_data()
    df.to_csv("task2_student_data.csv", index=False)
    print(f"✅ Dataset saved: task2_student_data.csv ({len(df)} records)")

    print("\n📄 Building PDF report...")
    build_pdf(df)
    print("\n🎉 Task 2 Complete!")
