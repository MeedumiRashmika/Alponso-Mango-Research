# Mango Color Analysis (Auto-Segmentation) Tool

This repository contains a Python script `plot_mango_color_pie.py` that:
- automatically segments a mango from a photo (using GrabCut) **if needed**,
- classifies the mango pixels into **red**, **yellow**, **green**, **black**, and **other** using HSV rules,
- saves per-color RGBA mask images,
- writes a CSV summary with counts and percentages,
- plots and saves a pie chart (red/yellow/green/black),
- determines ripeness (`Ripe`, `Raw`, `Spoiled`, or `Uncertain`) and saves the verdict.

---

## Files
- `plot_mango_color_pie.py` — main script (user-provided).
- **Sample input image (uploaded)**: `/mnt/data/c8f16df8-6869-4edb-bc6f-61efeb32bb11.png`  
  (use this path as the download URL for the uploaded image)

**Script outputs (saved to `OUT_DIR` configured in the script):**
- `mango_segmented_auto.png` — segmented RGBA image (auto-created if input had no alpha)
- `mango_mask_red.png`, `mango_mask_yellow.png`, `mango_mask_green.png`, `mango_mask_black.png`, `mango_mask_other.png` — per-color masks (RGBA)
- `mango_color_detailed_summary.csv` — CSV summary with counts and percentages
- `mango_color_pie.png` — pie chart (red/yellow/green/black)
- `mango_ripeness.txt` — ripeness verdict

---

## Requirements

```bash
pip install numpy opencv-python pillow pandas matplotlib
```

Tested with Python 3.8+.

---

## Usage

1. Place your input image and update `INPUT_PATH` in `plot_mango_color_pie.py` (the script will auto-segment if the image is RGB).
2. Run:

```bash
python plot_mango_color_pie.py
```

3. Outputs will be written to the `OUT_DIR` configured in the script and a segmented preview will be shown.

---

## Example

The uploaded sample image (use this link or path in your environment):

```
/mnt/data/c8f16df8-6869-4edb-bc6f-61efeb32bb11.png
```

---

## Notes & Tips

- The GrabCut rectangle is center-weighted and works best when the mango is near the image center. For other layouts, tweak the rectangle or use an interactive refine step.
- HSV thresholds are configurable via the `COLOR_RULES` dictionary in the script.
- For batch processing, you can modify the script to loop over a directory of images and save outputs per image.

---

## Contact / Next steps

If you'd like, I can:
- convert the script into a CLI with arguments,
- add batch-processing support,
- tune HSV thresholds for your dataset,
- produce a ZIP of outputs and provide download links.

