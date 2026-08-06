# Segmentation and Object Detection PCX Reproducibility Code

This repository contains PyTorch code for the experimental pipeline described in the paper. It focuses on explainability methods for semantic segmentation and object detection, with an emphasis on PCX/CRP-style concept-based explanations and LRP-based sanity checks.

## What is in this repository?

The codebase includes:

- PCX/CRP explanation utilities for segmentation and detection models
- PIDNet and YOLOv6 integration experiments
- LRP sanity-check scripts for verifying relevance conservation
- Support code for datasets such as flood segmentation, VOC2012, COCO2017, and person/car detection
- Supplementary experiment scripts under the [12-supp](12-supp) directory

## Repository layout

- [src](src): core explanation, plotting, helper, and dataset modules
- [configs](configs): configuration files for experiments
- [yolov6](yolov6): local YOLOv6 implementation and utilities
- [12-supp](12-supp): supplementary experiments and supporting code
- [LRP_sanity_check.py](LRP_sanity_check.py): standalone LRP sanity-check script
- [figure_pidnet_canonizer.py](figure_pidnet_canonizer.py): example PIDNet canonizer and visualization workflow
- [test_pidnet_canonizer.py](test_pidnet_canonizer.py): helper for testing PIDNet canonizer behavior
- [run_global_class_concepts.sh](run_global_class_concepts.sh): shell script entry point for selected concept-extraction experiments

## Environment

A Python environment with PyTorch is required. The repository includes a dependency list in [12-supp/code/requirements.txt](12-supp/code/requirements.txt).

Install dependencies with:

```bash
pip install -r 12-supp/code/requirements.txt
```

## Getting started

1. Clone the repository and enter the project root.
2. Install the required Python packages.
3. Adjust local model, checkpoint, and dataset paths in the scripts if needed.
4. Run the LRP sanity-check example:

```bash
python LRP_sanity_check.py --random_input --target_class 1 --y 32 --x 64
```

This command runs a lightweight sanity test with random input and checks whether relevance is conserved for a single target pixel.

## Final paper results

Two presentation notebooks summarize the saved experiment artifacts without rerunning training:

- [YOLO final results](notebooks/YOLO_final_results.ipynb)
- [PIDNet final results](notebooks/PIDNet_final_results.ipynb)

They look for `paper/12-supp/code/results` inside the checkout. If the result artifacts live elsewhere, start Jupyter with `PAPER_RESULTS_DIR=/path/to/results`.
Running either notebook exports its tables, generated plots, and selected figure artifacts under `paper output/yolo` or `paper output/pidnet`.

## Notes

- Several scripts contain hard-coded or environment-specific paths. You may need to edit them to match your local setup.
- The repository is research-oriented and may require a GPU for faster experimentation.
- The supplementary experiments in [12-supp/code/experiments](12-supp/code/experiments) are useful for reproducing additional paper results.

## Citation

Please refer to the paper itself for the correct citation details.
