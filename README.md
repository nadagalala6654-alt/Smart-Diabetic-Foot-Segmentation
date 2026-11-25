
# Smart Diabetic Foot Segmentation

## Project Overview
This project performs diabetic wound image segmentation using existing ground truth masks.

## Project Objectives
- Load wound images and their corresponding masks.
- Match images with their masks.
- Convert masks into binary masks.
- Generate segmentation overlays.
- Calculate wound area and percentage.
- Save segmentation results and reports.

## Project Structure
- Normal/
- wound_main/
- wound_mask/
- outputs/
  - binary_masks/
  - original_images/
  - overlays/
  - segmentation_preview.png
  - segmentation_report.csv
- segmentation.py
- requirements.txt
- README.md
- .gitignore

## Technologies
- Python
- OpenCV
- NumPy
- Matplotlib

## Important Note
This project uses existing ground truth masks only.

No AI model training is performed.

## Outputs
- Binary Masks
- Original Images
- Segmentation Overlays
- Segmentation Preview
- CSV Segmentation Report

## How to Run

Install the required libraries:

pip install -r requirements.txt

Run the segmentation script:

python segmentation.py
