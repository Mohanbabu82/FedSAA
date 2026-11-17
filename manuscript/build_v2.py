#!/usr/bin/env python3
"""Build the revised IEEE Access manuscript from content_v2.py into the IEEE Access Word template.

Usage: python build_v2.py <template.docx> <figures_dir> <output.docx>
"""
import copy
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import docx
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches
from docx.text.paragraph import Paragraph

import build_ieee as BI
import content_v2 as C

CITE = re.compile(r"\\cite\{([^}]*)\}")
XREF = re.compile(r"\{([TF]):([A-Za-z0-9_]+)\}")


def roman(n):
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def main():
    template, figdir, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    BODY, TABLES = C.build_body()

    # ------------------------------------------------ pass 1: order of tables, figures, citations
    tnum, fnum, cites = {}, {}, []

    def scan(text):
        for m in CITE.finditer(text):
            for k in m.group(1).split(","):
                k = k.strip()
                if k not in C.REFS:
                    raise KeyError("unknown citation " + k)
                if k not in cites:
                    cites.append(k)

    def place_table(k):
        tnum[k] = len(tnum) + 1
        spec = TABLES[k]
        for t in [spec["title"], spec.get("note") or ""] + [c for r in spec["rows"] for c in
                                                             (r[1:] if isinstance(r, tuple) else r)
                                                             if isinstance(c, str)]:
            scan(t)

    for it in BODY:
        k = it[0]
        if k in ("p", "pc", "h1", "h2", "h1u"):
            scan(it[1])
        elif k == "table":
            place_table(it[1])
        elif k == "wide_group":
            for t in it[1]:
                place_table(t)
        elif k == "fig":
            fnum[it[1]] = len(fnum) + 1
            scan(it[3])
        elif k == "alg":
            for _, line in it[1][1]:
                scan(line)
    unused = set(TABLES) - set(tnum)
    if unused:
        print("warning: tables defined but not placed:", unused)
    cnum = {k: i + 1 for i, k in enumerate(cites)}

    def fmt_cites(keys):
        ns = sorted(cnum[k.strip()] for k in keys.split(","))
        parts, i = [], 0
        while i < len(ns):
            j = i
            while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
                j += 1
            if j - i >= 2:
                parts.append(f"[{ns[i]}]–[{ns[j]}]")
            else:
                parts.extend(f"[{n}]" for n in ns[i:j + 1])
            i = j + 1
        return ", ".join(parts)

    def R(text):
        text = CITE.sub(lambda m: fmt_cites(m.group(1)), text)

        def xr(m):
            kind, key = m.group(1), m.group(2)
            if kind == "T":
                if key not in tnum:
                    raise KeyError("table not placed: " + key)
                return "Table " + roman(tnum[key])
            if key not in fnum:
                raise KeyError("figure not placed: " + key)
            return f"Fig. {fnum[key]}"
        return XREF.sub(xr, text)

    def spec_of(k):
        s = dict(TABLES[k])
        s["number"] = roman(tnum[k])
        s["title"] = R(s["title"])
        s["note"] = R(s["note"]) if s.get("note") else None
        rows = []
        for r in s["rows"]:
            if isinstance(r, tuple):
                rows.append((r[0], R(r[1]) if isinstance(r[1], str) else [R(x) for x in r[1]]))
            else:
                rows.append([R(x) for x in r])
        s["rows"] = rows
        s["header"] = [R(x) for x in s["header"]]
        return s

    # ------------------------------------------------ document set-up (as in build_ieee.main)
    doc = docx.Document(template)
    body = doc.element.body
    kids = list(body)
    final_sect = kids[-1]
    sect2 = copy.deepcopy(kids[17].find(qn("w:pPr")).find(qn("w:sectPr")))

    def sect(cols):
        s = copy.deepcopy(sect2)
        c = s.find(qn("w:cols"))
        if cols == 1:
            for a in list(c.attrib):
                del c.attrib[a]
            c.set(qn("w:space"), "720")
        return s

    BI.reset_runs(kids[2]); BI.add_runs(kids[2], C.TITLE)
    BI.reset_runs(kids[3]); BI.add_runs(kids[3], C.AUTHORS)
    body.remove(kids[4]); body.remove(kids[5])
    BI.reset_runs(kids[6]); BI.add_runs(kids[6], C.AFFILIATION)
    BI.reset_runs(kids[7]); BI.add_runs(kids[7], C.CORRESPONDING)
    BI.reset_runs(kids[8]); BI.add_runs(kids[8], C.SUPPORT)
    BI.reset_runs(kids[9], keep_first=1); BI.add_runs(kids[9], " " + R(C.ABSTRACT))
    BI.reset_runs(kids[10], keep_first=1); BI.add_runs(kids[10], " " + C.INDEX_TERMS)
    for el in kids[12:-1]:
        body.remove(el)

    numbering = doc.part.numbering_part.element
    used = [int(n.get(qn("w:numId"))) for n in numbering.findall(qn("w:num"))]

    def new_letter_list():
        nid = max(used) + 1
        used.append(nid)
        num = OxmlElement("w:num")
        num.set(qn("w:numId"), str(nid))
        a = OxmlElement("w:abstractNumId")
        a.set(qn("w:val"), "20")
        num.append(a)
        lo = OxmlElement("w:lvlOverride")
        lo.set(qn("w:ilvl"), "0")
        so = OxmlElement("w:startOverride")
        so.set(qn("w:val"), "1")
        lo.append(so)
        num.append(lo)
        numbering.findall(qn("w:num"))[-1].addnext(num)
        return nid

    eqs = [it[1] for it in BODY if it[0] == "eq"]
    maths = BI.latex_to_omml(eqs)
    M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    ORDER = {"dPr": ["begChr", "sepChr", "endChr", "grow", "shp", "ctrlPr"], "mcPr": ["count", "mcJc"]}
    for om in maths:
        for tag, order in ORDER.items():
            for el in om.iter(M + tag):
                kids_ = list(el)
                kids_.sort(key=lambda e: order.index(e.tag.replace(M, "")) if e.tag.replace(M, "") in order else 99)
                for e in kids_:
                    el.remove(e)
                    el.append(e)

    # ------------------------------------------------ figure and algorithm blocks
    def figure_block(key, fname, caption, width_in=3.15, total_w=BI.COL_W):
        tbl = parse_xml(
            f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{total_w}" w:type="dxa"/><w:jc w:val="center"/>'
            '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="0" w:type="dxa"/>'
            '<w:right w:w="0" w:type="dxa"/></w:tblCellMar><w:tblLook w:val="0000"/></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{total_w}"/></w:tblGrid></w:tbl>')
        for k in range(2):
            tr = OxmlElement("w:tr")
            trPr = OxmlElement("w:trPr")
            trPr.append(OxmlElement("w:cantSplit"))
            tr.append(trPr)
            tc = parse_xml(f'<w:tc {nsdecls("w")}><w:tcPr><w:tcW w:w="{total_w}" w:type="dxa"/></w:tcPr></w:tc>')
            if k == 0:
                pp = BI.mk_p(None, keep_next=True, jc="center", spacing={"before": 120, "after": 60},
                             ind={"left": 0, "firstLine": 0})
                Paragraph(pp, doc._body).add_run().add_picture(os.path.join(figdir, fname), width=Inches(width_in))
            else:
                pp = BI.mk_p("FigCaption", spacing={"before": 0, "after": 120}, ind={"left": 0, "firstLine": 0})
                BI.add_runs(pp, R(caption))
            tc.append(pp)
            tr.append(tc)
            tbl.append(tr)
        return [tbl, BI.mk_p("FootnoteText", spacing={"before": 0, "after": 80})]

    def alg_block(alg):
        title, lines = alg
        w = BI.COL_W
        tbl = parse_xml(
            f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{w}" w:type="dxa"/><w:jc w:val="center"/>'
            '<w:tblBorders>'
            '<w:top w:val="single" w:sz="8" w:space="0" w:color="auto"/>'
            '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="auto"/></w:tblBorders><w:tblLayout w:type="fixed"/>'
            '<w:tblCellMar><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar>'
            f'<w:tblLook w:val="0000"/></w:tblPr><w:tblGrid><w:gridCol w:w="{w}"/></w:tblGrid></w:tbl>')
        tr = OxmlElement("w:tr")
        trPr = OxmlElement("w:trPr")
        trPr.append(OxmlElement("w:cantSplit"))
        tr.append(trPr)
        tc = parse_xml(f'<w:tc {nsdecls("w")}><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/></w:tcPr></w:tc>')
        hp = BI.mk_p(None, spacing={"before": 40, "after": 40}, ind={"left": 0, "firstLine": 0}, keep_next=True)
        BI.add_runs(hp, title, size=16, bold=True)
        bd = OxmlElement("w:pBdr")
        b = OxmlElement("w:bottom")
        for a, v in (("val", "single"), ("sz", "4"), ("space", "1"), ("color", "auto")):
            b.set(qn("w:" + a), v)
        bd.append(b)
        hp.find(qn("w:pPr")).insert(1, bd)
        tc.append(hp)
        for lvl, line in lines:
            lp = BI.mk_p(None, spacing={"before": 0, "after": 0, "line": 220, "lineRule": "auto"},
                         ind={"left": 240 * lvl, "firstLine": 0}, keep_next=True)
            BI.add_runs(lp, R(line), size=16)
            tc.append(lp)
        tr.append(tc)
        tbl.append(tr)
        return [BI.mk_p("FootnoteText", spacing={"before": 60, "after": 0}), tbl,
                BI.mk_p("FootnoteText", spacing={"before": 0, "after": 140})]

    # ------------------------------------------------ body
    out = []
    st = {"first_h1": True, "after_heading": False, "h2_list": None, "h2_first": True, "eq": 0, "eqi": 0}

    def para(text, indent=True):
        style = "PARA" if (st["after_heading"] or not indent) else "PARAIndent"
        out.append(BI.mk_p(style, R(text)))
        st["after_heading"] = False

    def ensure_para_before():
        if not out or out[-1].tag != qn("w:p"):
            out.append(BI.mk_p("FootnoteText", spacing={"before": 0, "after": 0}))
        return out[-1]

    for it in BODY:
        k = it[0]
        if k == "h1":
            out.append(BI.mk_p("H1ListNoSpace" if st["first_h1"] else "H1ListSpace", it[1], keep_next=True))
            st.update(first_h1=False, after_heading=True, h2_list=new_letter_list(), h2_first=True)
        elif k == "h1u":
            out.append(BI.mk_p("H1", it[1], keep_next=True))
            st["after_heading"] = True
        elif k == "h2":
            style = "H2AfterH1" if (st["h2_first"] and st["after_heading"]) else "H2Cont"
            out.append(BI.mk_p(style, it[1], num_id=st["h2_list"], keep_next=True))
            st.update(h2_first=False, after_heading=True)
        elif k == "p":
            para(it[1])
        elif k == "pc":
            para(it[1], indent=False)
        elif k == "eq":
            st["eq"] += 1
            out.append(BI.mk_equation(maths[st["eqi"]], st["eq"]))
            st["eqi"] += 1
        elif k == "fig":
            out.extend(figure_block(it[1], it[2], it[3]))
        elif k == "table":
            out.extend(BI.table_block(spec_of(it[1])))
        elif k == "alg":
            out.extend(alg_block(it[1]))
        elif k == "wide_group":
            prev = ensure_para_before()
            prev.find(qn("w:pPr")).append(sect(2))
            block = []
            for t in it[1]:
                block.extend(BI.table_block(spec_of(t)))
            block[-1].find(qn("w:pPr")).append(sect(1))
            out.extend(block)
        else:
            raise ValueError(k)

    out.append(BI.mk_p("H1", "REFERENCES", keep_next=True))
    for key in cites:
        out.append(BI.mk_p("References", C.REFS[key]))
    for i, (name, text) in enumerate(C.BIOS):
        pp = BI.mk_p("AUBios", spacing={"before": 240 if i == 0 else 120})
        BI.add_runs(pp, name, bold=True)
        BI.add_runs(pp, text)
        out.append(pp)
    for el in out:
        final_sect.addprevious(el)

    cp = doc.core_properties
    cp.title = re.sub(r"[*~^\[\]]", "", C.TITLE)
    cp.author = "Mohan Babu A"
    cp.subject = "IEEE Access manuscript"
    cp.keywords = C.INDEX_TERMS
    doc.save(outpath)
    print(f"wrote {outpath}: {len(tnum)} tables, {len(fnum)} figures, {st['eq']} equations, {len(cites)} references")


if __name__ == "__main__":
    main()
