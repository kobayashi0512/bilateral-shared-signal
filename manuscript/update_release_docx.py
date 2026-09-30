#!/usr/bin/env python3
"""Create a release copy of the manuscript with the artifact URL and updated Figure 1."""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE


GITHUB_URL = "https://github.com/kobayashi0512/bilateral-shared-signal"
FIGURE1_CAPTION = (
    "Figure 1. Framework for conditional bilateral fundus correspondence. "
    "(A) DeepDRiD contributes 300 complete bilateral patients, with two views per eye; "
    "the four source photographs are illustrative. The patient-disjoint split is 210/46/44. "
    "(B) Same-patient left–right pairs are compared with cross-patient negatives; four cross-eye "
    "view combinations and same-eye repeat views audit view dependence. "
    "(C) Fixed ImageNet-pretrained ResNet-50 features are compared by cosine similarity; no "
    "representation is trained for this raw-feature measurement. "
    "(D) Negative assignment uses DR grade, exact grade × quality tier, and nearest measured "
    "quality within grade. Quality-composite differences of 0.1982, 0.0709, and 0.0240 indicate "
    "progressively tighter matching, not improving prediction. "
    "(E) DeepDRiD supplies internal evidence; CFP–OCT is a previously inspected development "
    "stress set; PAPILA is the locked external test of the frozen baseline only. Four DeepDRiD "
    "images and the PAPILA OD/OS pair are unmodified illustrative source photographs [10,11]."
)
DATA_AVAILABILITY = (
    "DeepDRiD [10] and PAPILA [11] are available from their cited official sources; CFP–OCT is "
    "available from the source repository cited in [24]."
)
PROJECT_MATERIALS = (
    "The PAPILA evaluation used an audited Kaggle distribution copy of the official dataset, "
    "not a separate cohort. Analysis scripts, frozen protocols, identity-free figure inputs, "
    "aggregate result reports, and editable figure sources are publicly available at "
)
PROJECT_MATERIALS_TAIL = (
    ". Full source archives, patient-level manifests and split assignments, clinical tables, "
    "feature caches, and model checkpoints are not redistributed. Selected illustrative source "
    "photographs are embedded in Figures 1 and 4 under their respective dataset terms."
)


def replace_paragraph(paragraph, label: str, body: str | None = None) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)
    if label:
        run = paragraph.add_run(label)
        run.bold = True
    if body:
        paragraph.add_run(body)


def append_hyperlink(paragraph, text: str, url: str) -> None:
    rel_id = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    style = OxmlElement("w:rStyle")
    style.set(qn("w:val"), "Hyperlink")
    props.append(style)
    run.append(props)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--figure1-preview", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise SystemExit("Refusing to overwrite the source manuscript.")

    doc = Document(args.source)
    figure_caption = next((p for p in doc.paragraphs if p.text.strip().startswith("Figure 1.")), None)
    data_paragraph = next((p for p in doc.paragraphs if p.text.strip().startswith("Data availability.")), None)
    declarations_heading = next((p for p in doc.paragraphs if p.text.strip() == "Declarations and reproducibility status"), None)
    ethics_paragraph = next((p for p in doc.paragraphs if p.text.strip().startswith("Ethics statement.")), None)
    if figure_caption is None or data_paragraph is None or declarations_heading is None or ethics_paragraph is None:
        raise SystemExit("Required Figure 1 caption or Data availability paragraph was not found.")
    if not doc.inline_shapes:
        raise SystemExit("No embedded figures found; cannot replace Figure 1 safely.")

    first_shape = doc.inline_shapes[0]
    rel_id = first_shape._inline.graphic.graphicData.pic.blipFill.blip.embed
    image_part = doc.part.related_parts[rel_id]
    image_part._blob = args.figure1_preview.read_bytes()

    replace_paragraph(figure_caption, "Figure 1. ", FIGURE1_CAPTION.removeprefix("Figure 1. "))
    replace_paragraph(data_paragraph, "Data availability. ", DATA_AVAILABILITY)
    details = ethics_paragraph.insert_paragraph_before()
    details.style = data_paragraph.style
    label = details.add_run("Project materials. ")
    label.bold = True
    details.add_run(PROJECT_MATERIALS)
    append_hyperlink(details, GITHUB_URL, GITHUB_URL)
    details.add_run(PROJECT_MATERIALS_TAIL)
    declarations_heading.paragraph_format.keep_with_next = True
    details.paragraph_format.keep_together = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.output)
    print(f"Wrote {args.output}")
    print(f"Updated Figure 1 media relationship {rel_id}; added hyperlink {GITHUB_URL}")


if __name__ == "__main__":
    main()
