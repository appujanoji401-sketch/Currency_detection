from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from pathlib import Path
import re

def parse_markdown(md_file):
    """Parse markdown file into structured content"""
    with open(md_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    content = []
    current_list = None
    code_block = False
    
    for line in lines:
        line = line.rstrip()
        
        # Code blocks
        if line.startswith('```'):
            if code_block:
                content.append(('code_end',))
                code_block = False
            else:
                content.append(('code_start',))
                code_block = True
            continue
        
        if code_block:
            content.append(('code_line', line))
            continue
        
        # Headers
        if line.startswith('# '):
            content.append(('h1', line[2:]))
        elif line.startswith('## '):
            content.append(('h2', line[3:]))
        elif line.startswith('### '):
            content.append(('h3', line[4:]))
        elif line.startswith('#### '):
            content.append(('h4', line[5:]))
        # Lists
        elif line.startswith('- ') or line.startswith('* '):
            if current_list != 'ul':
                content.append(('ul_start',))
                current_list = 'ul'
            content.append(('li', line[2:]))
        elif re.match(r'^\d+\. ', line):
            if current_list != 'ol':
                content.append(('ol_start',))
                current_list = 'ol'
            content.append(('li', re.sub(r'^\d+\. ', '', line)))
        # Tables
        elif '|' in line and line.strip().startswith('|'):
            content.append(('table_row', line))
        # Horizontal rules
        elif line.strip() == '---':
            content.append(('hr',))
        # Empty lines end lists
        elif not line.strip():
            if current_list:
                content.append(('list_end',))
                current_list = None
            content.append(('spacer',))
        # Regular paragraphs
        else:
            if current_list:
                content.append(('list_end',))
                current_list = None
            content.append(('p', line))
    
    if current_list:
        content.append(('list_end',))
    
    return content

def create_pdf(md_file, pdf_file):
    """Create PDF from markdown file"""
    
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        spaceBefore=30,
        alignment=TA_CENTER
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        spaceAfter=12,
        leading=14
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=9,
        textColor=colors.HexColor('#333333'),
        backColor=colors.HexColor('#f4f4f4'),
        leftIndent=20,
        spaceAfter=10
    )
    
    # Parse content
    content = parse_markdown(md_file)
    story = []
    
    for item in content:
        item_type = item[0]
        
        if item_type == 'h1':
            story.append(Paragraph(item[1], title_style))
        elif item_type == 'h2':
            story.append(Paragraph(item[1], h2_style))
        elif item_type == 'h3':
            story.append(Paragraph(item[1], h3_style))
        elif item_type == 'p':
            text = item[1].replace('**', '<b>').replace('**', '</b>')
            text = text.replace('*', '<i>').replace('*', '</i>')
            text = text.replace('`', '<font face="Courier">').replace('`', '</font>')
            story.append(Paragraph(text, body_style))
        elif item_type == 'code_start':
            story.append(Spacer(1, 0.1*inch))
        elif item_type == 'code_line':
            story.append(Paragraph(item[1], code_style))
        elif item_type == 'code_end':
            story.append(Spacer(1, 0.1*inch))
        elif item_type == 'spacer':
            story.append(Spacer(1, 0.1*inch))
        elif item_type == 'hr':
            story.append(Spacer(1, 0.2*inch))
        elif item_type == 'ul_start':
            pass
        elif item_type == 'ol_start':
            pass
        elif item_type == 'li':
            text = f"• {item[1]}"
            story.append(Paragraph(text, body_style))
        elif item_type == 'list_end':
            story.append(Spacer(1, 0.1*inch))
        elif item_type == 'table_row':
            cells = [cell.strip() for cell in item[1].split('|')]
            cells = [c for c in cells if c]  # Remove empty cells
            if len(cells) > 1:
                table = Table([cells], colWidths=[3*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
    
    doc.build(story)
    print(f"PDF generated successfully: {pdf_file}")

if __name__ == "__main__":
    md_file = Path("PROJECT_ANALYSIS.md")
    pdf_file = Path("PROJECT_ANALYSIS.pdf")
    
    create_pdf(md_file, pdf_file)
