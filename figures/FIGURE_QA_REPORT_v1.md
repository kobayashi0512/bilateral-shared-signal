# Main-figure quality and evidence audit

- Figure sequence, captions, and file numbering are aligned: framework (1), hierarchy (2), conditional controls (3), external evidence (4).
- Figure 2 was regenerated from the frozen patient-level hierarchy CSV and the five seedwise 3,000-resample patient-bootstrap CIs. The test cohort has 43 patients; validation has 45. For both splits, every same-eye-minus-cross-eye CI includes zero, while every cross-eye-minus-grade-matched-negative CI is strictly positive. Test means are 0.8308, 0.8241, and 0.7366 for the three similarity levels, respectively.
- Figure 3's patient-bootstrap CI envelopes were read from the frozen bootstrap CSV, not inferred from across-seed SD. Envelope lower bounds for the three controls are 0.8479, 0.8275, and 0.8181. The plotted AUROC axis is intentionally enlarged; chance 0.5 is explicitly marked as off-scale.
- Figure 4 repeats the locked PAPILA frozen baseline; no H-CBFD score is assigned to PAPILA. Its caption defines OS/OD and the two CFP–OCT DME label conventions.
- Export QA: each quantitative figure is 180 mm wide, with Arial embedded in PDF, editable text in SVG, and a 600-dpi LZW TIFF. Figure 1 is separately checked as editable PPTX/SVG with four original DeepDRiD and two original PAPILA photographs and native text/routes.
- Visual QA: inspected the final PNG previews for truncation, overlap, axis labels, legends, and grayscale-readable marker shapes. Figure 2's earlier long title and axis/footer collisions were corrected before delivery.
- Open before submission: target-journal final figure dimensions and verification of the combined-figure image reuse/citation obligations under both source licenses. These are policy checks, not missing computational results.
