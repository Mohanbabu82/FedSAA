#!/usr/bin/env python3
"""Fill the IEEE Access Word template with the manuscript in content.py.

Usage:
    python build_ieee.py <template.docx> <figures_dir> <output.docx>

The template's own styles, numbering, headers, footers and first-page layout are kept.
The template's instructional body text is removed and replaced.
"""
import copy
import os
import re
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import docx
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches
from docx.text.paragraph import Paragraph
from lxml import etree

import content as C

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
COL_W = 4820       # one text column, twips
TOK = re.compile(r"(\*\*|\*|~|\^|\[\[|\]\])")


# ---------------------------------------------------------------- runs and paragraphs
def add_runs(p, text, size=None, bold=False, italic=False):
    """Append runs to paragraph element p, interpreting the inline markup."""
    b = i = sub = sup = hl = False
    for tok in TOK.split(text):
        if tok == "**":
            b = not b
        elif tok == "*":
            i = not i
        elif tok == "~":
            sub = not sub
        elif tok == "^":
            sup = not sup
        elif tok == "[[":
            hl = True
        elif tok == "]]":
            hl = False
        elif tok:
            r = OxmlElement("w:r")
            rPr = OxmlElement("w:rPr")
            if b or bold:
                rPr.append(OxmlElement("w:b"))
            if i or italic:
                rPr.append(OxmlElement("w:i"))
            if size:
                for tag in ("w:sz", "w:szCs"):
                    e = OxmlElement(tag)
                    e.set(qn("w:val"), str(size))
                    rPr.append(e)
            if hl:
                e = OxmlElement("w:highlight")
                e.set(qn("w:val"), "yellow")
                rPr.append(e)
            if sub or sup:
                e = OxmlElement("w:vertAlign")
                e.set(qn("w:val"), "subscript" if sub else "superscript")
                rPr.append(e)
            if len(rPr):
                r.append(rPr)
            t = OxmlElement("w:t")
            t.set(qn("xml:space"), "preserve")
            t.text = tok
            r.append(t)
            p.append(r)


def mk_p(style, text=None, *, num_id=None, keep_next=False, jc=None, spacing=None, ind=None,
         tabs=None, size=None):
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    p.append(pPr)
    if style:
        e = OxmlElement("w:pStyle")
        e.set(qn("w:val"), style)
        pPr.append(e)
    if keep_next:
        pPr.append(OxmlElement("w:keepNext"))
    if num_id is not None:
        numPr = OxmlElement("w:numPr")
        a = OxmlElement("w:ilvl")
        a.set(qn("w:val"), "0")
        b = OxmlElement("w:numId")
        b.set(qn("w:val"), str(num_id))
        numPr.extend([a, b])
        pPr.append(numPr)
    if tabs:
        t = OxmlElement("w:tabs")
        for val, pos in tabs:
            e = OxmlElement("w:tab")
            e.set(qn("w:val"), val)
            e.set(qn("w:pos"), str(pos))
            t.append(e)
        pPr.append(t)
    if spacing:
        e = OxmlElement("w:spacing")
        for k, v in spacing.items():
            e.set(qn("w:" + k), str(v))
        pPr.append(e)
    if ind:
        e = OxmlElement("w:ind")
        for k, v in ind.items():
            e.set(qn("w:" + k), str(v))
        pPr.append(e)
    if jc:
        e = OxmlElement("w:jc")
        e.set(qn("w:val"), jc)
        pPr.append(e)
    if text:
        add_runs(p, text, size=size)
    return p


def reset_runs(p, keep_first=0):
    """Remove everything in p except pPr and its first keep_first runs."""
    kept = 0
    for child in list(p):
        if child.tag == qn("w:pPr"):
            continue
        if child.tag == qn("w:r") and kept < keep_first:
            kept += 1
            continue
        p.remove(child)


# ---------------------------------------------------------------- equations
def latex_to_omml(latex_list):
    """Convert display equations to Office Math with pandoc; returns a list of m:oMath elements."""
    with tempfile.TemporaryDirectory() as tmp:
        md = os.path.join(tmp, "eq.md")
        out = os.path.join(tmp, "eq.docx")
        with open(md, "w", encoding="utf-8") as fh:
            fh.write("\n\n".join("$$" + s + "$$" for s in latex_list) + "\n")
        subprocess.run(["pandoc", "-f", "markdown", "-t", "docx", "-o", out, md], check=True)
        root = etree.fromstring(zipfile.ZipFile(out).read("word/document.xml"))
    maths = [copy.deepcopy(e) for e in root.iter("{%s}oMath" % M_NS)]
    if len(maths) != len(latex_list):
        raise RuntimeError(f"pandoc produced {len(maths)} equations for {len(latex_list)} inputs")
    return maths


def mk_equation(omath, number):
    p = mk_p("Equation", tabs=[("clear", 4584), ("center", COL_W // 2), ("right", COL_W)],
             ind={"left": 0, "firstLine": 0}, jc="left")
    r = OxmlElement("w:r")
    r.append(OxmlElement("w:tab"))
    p.append(r)
    p.append(omath)
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    e = OxmlElement("w:i")
    e.set(qn("w:val"), "0")
    rPr.append(e)
    r.append(rPr)
    r.append(OxmlElement("w:tab"))
    t = OxmlElement("w:t")
    t.text = f"({number})"
    r.append(t)
    p.append(r)
    return p


# ---------------------------------------------------------------- tables
def _borders(top=None, bottom=None):
    if not (top or bottom):
        return ""
    s = "<w:tcBorders>"
    if top:
        s += f'<w:top w:val="{top[0]}" w:sz="{top[1]}" w:space="0" w:color="auto"/>'
    if bottom:
        s += f'<w:bottom w:val="{bottom[0]}" w:sz="{bottom[1]}" w:space="0" w:color="auto"/>'
    return s + "</w:tcBorders>"


def _cell(text, width, align, size, span=1, top=None, bottom=None, italic=False, bold=False):
    tc = parse_xml(
        f'<w:tc {nsdecls("w")}><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>'
        + (f'<w:gridSpan w:val="{span}"/>' if span > 1 else "")
        + _borders(top, bottom)
        + '<w:vAlign w:val="center"/></w:tcPr></w:tc>'
    )
    p = mk_p(None, spacing={"before": 20, "after": 20}, jc=align)
    add_runs(p, text, size=size, italic=italic, bold=bold)
    tc.append(p)
    return tc


def mk_table(spec):
    widths, size, aligns = spec["widths"], spec["size"], spec["align"]
    n = len(widths)
    tbl = parse_xml(
        f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/><w:jc w:val="center"/>'
        '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="50" w:type="dxa"/>'
        '<w:right w:w="50" w:type="dxa"/></w:tblCellMar><w:tblLook w:val="0000"/></w:tblPr><w:tblGrid>'
        + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        + "</w:tblGrid></w:tbl>"
    )

    def row(cells):
        tr = OxmlElement("w:tr")
        trPr = OxmlElement("w:trPr")
        trPr.append(OxmlElement("w:cantSplit"))
        tr.append(trPr)
        for c in cells:
            tr.append(c)
        tbl.append(tr)

    row([_cell(h, w, "center", size, top=("double", 6), bottom=("single", 6))
         for h, w in zip(spec["header"], widths)])
    rows = spec["rows"]
    for k, r in enumerate(rows):
        last = k == len(rows) - 1
        bottom = ("single", 8) if last else None
        if isinstance(r, tuple) and r[0] == "group":
            top = ("single", 4) if k > 0 else None
            row([_cell(r[1], sum(widths), "left", size, span=n, top=top, bottom=bottom, italic=True)])
        elif isinstance(r, tuple) and r[0] == "header":
            row([_cell(h, w, "center", size, top=("single", 6), bottom=("single", 6))
                 for h, w in zip(r[1], widths)])
        else:
            row([_cell(t, w, a, size, bottom=bottom) for t, w, a in zip(r, widths, aligns)])
    return tbl


def table_block(spec):
    els = [mk_p("TableTitle", "TABLE " + spec["number"], keep_next=True, spacing={"before": 160}),
           mk_p("TableTitle", spec["title"], keep_next=True, spacing={"after": 60}),
           mk_table(spec)]
    if spec.get("note"):
        els.append(mk_p("FootnoteText", spec["note"], spacing={"before": 40, "after": 200}))
    else:
        els.append(mk_p("FootnoteText", spacing={"before": 0, "after": 140}))
    return els


# ---------------------------------------------------------------- main
def main():
    template, figdir, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    doc = docx.Document(template)
    body = doc.element.body
    kids = list(body)
    final_sect = kids[-1]
    assert final_sect.tag == qn("w:sectPr")

    # section-properties prototype for a continuous two-column section (template paragraph 17)
    sect2 = copy.deepcopy(kids[17].find(qn("w:pPr")).find(qn("w:sectPr")))

    def sect(cols):
        s = copy.deepcopy(sect2)
        c = s.find(qn("w:cols"))
        if cols == 1:
            for a in list(c.attrib):
                del c.attrib[a]
            c.set(qn("w:space"), "720")
        return s

    # ---- front matter (template paragraphs 0-11 are the single-column first section)
    reset_runs(kids[2]);  add_runs(kids[2], C.TITLE)
    reset_runs(kids[3]);  add_runs(kids[3], C.AUTHORS)
    body.remove(kids[4]); body.remove(kids[5])
    reset_runs(kids[6]);  add_runs(kids[6], C.AFFILIATION)
    reset_runs(kids[7]);  add_runs(kids[7], C.CORRESPONDING)
    reset_runs(kids[8]);  add_runs(kids[8], C.SUPPORT)
    reset_runs(kids[9], keep_first=1);  add_runs(kids[9], " " + C.ABSTRACT)
    reset_runs(kids[10], keep_first=1); add_runs(kids[10], " " + C.INDEX_TERMS)

    # ---- remove the template's instructional body
    for el in kids[12:-1]:
        body.remove(el)

    # ---- numbering: one fresh lettered list per section so subsections restart at A
    numbering = doc.part.numbering_part.element
    used = [int(n.get(qn("w:numId"))) for n in numbering.findall(qn("w:num"))]

    def new_letter_list():
        nid = max(used) + 1
        used.append(nid)
        num = OxmlElement("w:num")
        num.set(qn("w:numId"), str(nid))
        a = OxmlElement("w:abstractNumId")
        a.set(qn("w:val"), "20")           # the template's "A." list used by H2_After H1
        num.append(a)
        lo = OxmlElement("w:lvlOverride")
        lo.set(qn("w:ilvl"), "0")
        so = OxmlElement("w:startOverride")
        so.set(qn("w:val"), "1")
        lo.append(so)
        num.append(lo)
        numbering.findall(qn("w:num"))[-1].addnext(num)
        return nid

    maths = latex_to_omml(C.EQUATIONS)
    out = []                 # elements to insert, in order
    state = {"first_h1": True, "after_heading": False, "h2_list": None, "h2_first": True, "eq": 0}

    def para(text, indent=True):
        style = "PARA" if (state["after_heading"] or not indent) else "PARAIndent"
        out.append(mk_p(style, text))
        state["after_heading"] = False

    for item in C.BODY:
        kind = item[0]
        if kind == "h1":
            out.append(mk_p("H1ListNoSpace" if state["first_h1"] else "H1ListSpace", item[1], keep_next=True))
            state.update(first_h1=False, after_heading=True, h2_list=new_letter_list(), h2_first=True)
        elif kind == "h1u":
            out.append(mk_p("H1", item[1], keep_next=True))
            state["after_heading"] = True
        elif kind == "h2":
            style = "H2AfterH1" if (state["h2_first"] and state["after_heading"]) else "H2Cont"
            out.append(mk_p(style, item[1], num_id=state["h2_list"], keep_next=True))
            state.update(h2_first=False, after_heading=True)
        elif kind == "p":
            para(item[1])
        elif kind == "pc":
            para(item[1], indent=False)
        elif kind == "eq":
            state["eq"] += 1
            out.append(mk_equation(maths[item[1]], state["eq"]))
        elif kind == "fig":
            p = mk_p("FigureCaption", keep_next=True, spacing={"before": 160}, jc="center")
            Paragraph(p, doc._body).add_run().add_picture(os.path.join(figdir, item[1]), width=Inches(3.3))
            out.append(p)
            out.append(mk_p("FigCaption", item[2], spacing={"after": 200}))
        elif kind == "table":
            out.extend(table_block(C.TABLES[item[1]]))
        elif kind == "table_wide":
            # end the two-column section on the preceding paragraph, set the table across the
            # full page in a one-column section, then resume two columns
            prev = out[-1]
            assert prev.tag == qn("w:p")
            prev.find(qn("w:pPr")).append(sect(2))
            block = table_block(C.TABLES[item[1]])
            block[-1].find(qn("w:pPr")).append(sect(1))
            out.extend(block)
        else:
            raise ValueError(kind)

    # ---- references and biographies
    out.append(mk_p("H1", "REFERENCES", keep_next=True))
    for ref in C.REFERENCES:
        out.append(mk_p("References", ref))
    for k, (name, text) in enumerate(C.BIOS):
        p = mk_p("AUBios", spacing={"before": 240 if k == 0 else 120})
        add_runs(p, name, bold=True)
        add_runs(p, text)
        out.append(p)

    for el in out:
        final_sect.addprevious(el)

    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    doc.save(outpath)
    print("wrote", outpath, "| body elements:", len(body), "| equations:", state["eq"],
          "| references:", len(C.REFERENCES))


if __name__ == "__main__":
    main()
