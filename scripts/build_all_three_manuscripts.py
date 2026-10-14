#!/usr/bin/env python3
"""
build_all_three_manuscripts.py
Script Komprehensif untuk Menghasilkan 3 Naskah Jurnal Ilmiah Siap Submit & Terbit
Format: Template Resmi MIND Journal (Terakreditasi SINTA 3 Kemendikbudristek)
Penulis: Maheza Novrayuda
Afiliasi: Program Studi Teknik Informatika, Fakultas Ilmu Komputer, Universitas Putra Indonesia YPTK Padang
Email: maheza.novrayuda@upiyptk.ac.id
"""

import os
import sys
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_base_document():
    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(1.5748)     # 4.0 cm
        section.bottom_margin = Inches(1.5748)  # 4.0 cm
        section.left_margin = Inches(1.5748)    # 4.0 cm
        section.right_margin = Inches(1.5748)   # 4.0 cm
        section.page_width = Inches(8.267)      # A4
        section.page_height = Inches(11.692)
    return doc

def set_run_font(run, font_name="Tahoma", size_pt=11, bold=False, italic=False, color_rgb=(0,0,0)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)

def add_p(doc, text="", font_name="Tahoma", size_pt=11, bold=False, italic=False, 
          color_rgb=(0,0,0), align=WD_ALIGN_PARAGRAPH.JUSTIFY, 
          space_before=0, space_after=4, line_spacing=1.15):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        r = p.add_run(text)
        set_run_font(r, font_name, size_pt, bold, italic, color_rgb)
    return p

def add_heading_1(doc, text):
    return add_p(doc, text, size_pt=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=12, space_after=4)

def add_heading_2(doc, text):
    return add_p(doc, text, size_pt=11, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=8, space_after=3)

def add_heading_3(doc, text):
    return add_p(doc, text, size_pt=11, bold=True, italic=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=6, space_after=2)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="12" w:space="0" w:color="222222"/>'
        f'  <w:bottom w:val="single" w:sz="12" w:space="0" w:color="222222"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="DDDDDD"/>'
        f'  <w:insideV w:val="none"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_table_data(doc, headers, rows, col_widths=None):
    tbl = doc.add_table(rows=len(rows)+1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl)
    
    # Header Row
    hdr_cells = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h)
        set_run_font(r, font_name="Tahoma", size_pt=9.5, bold=True, color_rgb=(0,0,0))
        set_cell_background(hdr_cells[i], "F0F4F8")
        set_cell_margins(hdr_cells[i])
        hdr_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Data Rows
    for r_idx, row_data in enumerate(rows):
        row_cells = tbl.rows[r_idx+1].cells
        bg_col = "FFFFFF" if r_idx % 2 == 0 else "F9FBFC"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = ""
            p = row_cells[c_idx].paragraphs[0]
            # align left for first column, center for others
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(str(val))
            set_run_font(r, font_name="Tahoma", size_pt=9.5, bold=(r_idx == len(rows)-1 and "Proposed" in str(row_data[0])))
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx])
            row_cells[c_idx].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    if col_widths:
        for row in tbl.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return tbl

def compress_img(path, quality=85, max_w=1200):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if img.width > max_w:
        w_p = (max_w / float(img.width))
        h_s = int((float(img.height) * float(w_p)))
        img = img.resize((max_w, h_s), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality, optimize=True)
    buf.seek(0)
    return buf

def add_image_figure(doc, img_path, caption_text, width_inches=5.1):
    c_buf = compress_img(img_path)
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.add_run().add_picture(c_buf, width=Inches(width_inches))
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(8)
    r_cap = p_cap.add_run(caption_text)
    set_run_font(r_cap, font_name="Tahoma", size_pt=9.5, italic=True)

if __name__ == '__main__':
    print("=== Membangun 3 Manuskrip Jurnal Ilmiah (MIND Journal SINTA 3) ===")
    from scripts.manuscript_jurnal_1 import build_jurnal_1
    from scripts.manuscript_jurnal_2 import build_jurnal_2
    from scripts.manuscript_jurnal_3 import build_jurnal_3
    
    os.makedirs('Jurnal 1', exist_ok=True)
    os.makedirs('Jurnal 2', exist_ok=True)
    os.makedirs('Jurnal 3', exist_ok=True)
    
    build_jurnal_1('Jurnal 1/Draf_Jurnal_1_YOLOv8_P2_CBAM_WIoU_MIND_SINTA3.docx')
    build_jurnal_2('Jurnal 2/Draf_Jurnal_2_XAI_YOLOv8_CBAM_MIND_SINTA3.docx')
    build_jurnal_3('Jurnal 3/Draf_Jurnal_3_Kinerja_Efisiensi_Real_Time_YOLOv8_UAV_MIND_SINTA3.docx')
    print("=== Seluruh 3 Manuskrip Berhasil Dibangun dan Tervalidasi ===")

