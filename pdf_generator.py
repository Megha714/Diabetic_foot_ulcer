"""
PDF Report Generator for DFU Detection System
Generates comprehensive health reports with patient data and predictions
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import os

def generate_patient_report(patient, predictions, output_path):
    """
    Generate a comprehensive PDF report for a patient
    
    Args:
        patient (dict): Patient information
        predictions (list): List of prediction results
        output_path (str): Path where PDF will be saved
    
    Returns:
        str: Path to generated PDF file
    """
    
    # Create the PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    normal_style = styles['Normal']
    
    # Title
    title = Paragraph("🏥 Diabetic Foot Ulcer Detection Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Report metadata
    report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    metadata = Paragraph(f"<b>Report Generated:</b> {report_date}", normal_style)
    elements.append(metadata)
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#667eea')))
    elements.append(Spacer(1, 20))
    
    # Patient Information Section
    patient_heading = Paragraph("📋 Patient Information", heading_style)
    elements.append(patient_heading)
    
    patient_data = [
        ['Patient ID:', patient.get('patient_id', 'N/A')],
        ['Name:', patient.get('name', patient.get('full_name', 'N/A'))],
        ['Age:', f"{patient.get('age', 'N/A')} years"],
        ['Gender:', patient.get('gender', 'N/A')],
        ['Contact:', patient.get('contact', patient.get('phone', 'N/A'))],
        ['Registration Date:', patient.get('created_at', datetime.now()).strftime('%Y-%m-%d') if patient.get('created_at') else 'N/A']
    ]
    
    # Add diabetes information if available
    if patient.get('has_diabetes'):
        patient_data.extend([
            ['', ''],
            ['Diabetes Status:', 'Yes'],
            ['Diabetes Type:', patient.get('diabetes_type', 'N/A')],
            ['Diabetes Duration:', f"{patient.get('diabetes_duration', 'N/A')} years"],
            ['Blood Sugar Level:', f"{patient.get('blood_sugar_level', 'N/A')} mg/dL"],
            ['HbA1c Level:', f"{patient.get('hba1c_level', 'N/A')}%"]
        ])
    else:
        patient_data.append(['Diabetes Status:', 'No'])
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9ff')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#333333')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    
    elements.append(patient_table)
    elements.append(Spacer(1, 20))
    
    # Health Check Summary
    summary_heading = Paragraph("📊 Health Check Summary", heading_style)
    elements.append(summary_heading)
    
    total_checks = len(predictions)
    healthy_count = sum(1 for p in predictions if p.get('prediction_class') == 0)
    ulcer_count = sum(1 for p in predictions if p.get('prediction_class') == 1)
    rejected_count = sum(1 for p in predictions if p.get('rejected'))
    
    summary_data = [
        ['Total Health Checks:', str(total_checks)],
        ['Healthy Results:', str(healthy_count)],
        ['Ulcers Detected:', str(ulcer_count)],
        ['Invalid Images:', str(rejected_count)]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#333333')),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#667eea'))
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    # Detailed Predictions
    if predictions:
        predictions_heading = Paragraph("🔬 Detailed Health Check Results", heading_style)
        elements.append(predictions_heading)
        elements.append(Spacer(1, 12))
        
        for idx, pred in enumerate(predictions, 1):
            # Prediction header
            pred_date = pred.get('prediction_date', datetime.now())
            if isinstance(pred_date, datetime):
                pred_date_str = pred_date.strftime('%B %d, %Y at %I:%M %p')
            else:
                pred_date_str = str(pred_date)
            
            pred_title = Paragraph(f"<b>Health Check #{idx}</b> - {pred_date_str}", normal_style)
            elements.append(pred_title)
            elements.append(Spacer(1, 6))
            
            # Result details
            if pred.get('rejected'):
                result_color = colors.grey
                result_text = "❌ Invalid Image"
                diagnosis = pred.get('rejection_reason', 'Not a foot image')
            elif pred.get('prediction_class') == 0:
                result_color = colors.green
                result_text = "✅ Healthy"
                diagnosis = "Normal (Healthy skin)"
            else:
                result_color = colors.red
                result_text = "⚠️ Ulcer Detected"
                diagnosis = "Abnormal (Ulcer detected)"
            
            confidence = pred.get('confidence', 0) * 100 if pred.get('confidence') else 0
            
            pred_data = [
                ['Result:', result_text],
                ['Diagnosis:', diagnosis],
                ['Confidence:', f"{confidence:.2f}%"]
            ]
            
            pred_table = Table(pred_data, colWidths=[1.5*inch, 4.5*inch])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9ff')),
                ('TEXTCOLOR', (1, 0), (1, 0), result_color),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            
            elements.append(pred_table)
            elements.append(Spacer(1, 15))
    
    else:
        no_data = Paragraph("<i>No health check results available</i>", normal_style)
        elements.append(no_data)
    
    # Footer
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    footer_text = Paragraph(
        "<i>This report is generated by the DFU Detection System using AI-powered Vision Transformer technology. "
        "For medical advice, please consult with a healthcare professional.</i>",
        ParagraphStyle('Footer', parent=normal_style, fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    )
    elements.append(Spacer(1, 6))
    elements.append(footer_text)
    
    # Build PDF
    doc.build(elements)
    
    return output_path
