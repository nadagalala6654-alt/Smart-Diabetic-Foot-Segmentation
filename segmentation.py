
import os
import re
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SETTINGS
# ==========================================

IMAGE_FOLDER = "wound_main"
MASK_FOLDER = "wound_mask"
OUTPUT_FOLDER = "outputs"

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ==========================================
# 2. CREATE OUTPUT FOLDERS
# ==========================================

folders = [
    "original_images",
    "binary_masks",
    "overlays"
]

for folder in folders:
    os.makedirs(
        os.path.join(OUTPUT_FOLDER, folder),
        exist_ok=True
    )

# ==========================================
# 3. LOAD DATASET
# ==========================================

images = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(EXTENSIONS)
])

masks = sorted([
    f for f in os.listdir(MASK_FOLDER)
    if f.lower().endswith(EXTENSIONS)
])

print("========== DATASET INFORMATION ==========")

print("Wound Images:", len(images))
print("Wound Masks:", len(masks))

# ==========================================
# 4. MATCH IMAGES AND MASKS
# ==========================================

def get_file_number(filename):

    name = os.path.splitext(filename)[0]

    numbers = re.findall(r'\d+', name)

    if numbers:
        return numbers[-1]

    return None


mask_dict = {}

for mask_name in masks:

    number = get_file_number(mask_name)

    if number is not None:

        if number in mask_dict:
            raise ValueError(
                "Duplicate mask number: " + number
            )

        mask_dict[number] = mask_name


matched_files = []
unmatched_images = []

for image_name in images:

    number = get_file_number(image_name)

    if number is not None and number in mask_dict:

        matched_files.append(
            (image_name, mask_dict[number])
        )

    else:
        unmatched_images.append(image_name)


print("\n========== MATCHING RESULTS ==========")

print("Matched Pairs:", len(matched_files))
print("Unmatched Images:", len(unmatched_images))

if len(matched_files) == 0:

    raise ValueError(
        "No matching image-mask pairs found!"
    )

# ==========================================
# 5. PROCESS IMAGES AND MASKS
# ==========================================

report = []

preview_items = []

processed = 0
failed = 0
empty_masks = 0

for index, (image_name, mask_name) in enumerate(matched_files):

    image_path = os.path.join(
        IMAGE_FOLDER, image_name
    )

    mask_path = os.path.join(
        MASK_FOLDER, mask_name
    )

    # Load original image
    image = cv2.imread(image_path)

    # Load mask
    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:

        print("Error reading:", image_name)

        failed += 1
        continue

    # Resize mask to match image dimensions
    if image.shape[:2] != mask.shape[:2]:

        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Convert original image to RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Create binary mask
    binary_mask = (
        mask > 0
    ).astype(np.uint8) * 255

    # Create overlay
    overlay = image_rgb.copy()

    overlay[binary_mask > 0] = [255, 0, 0]

    # ======================================
    # 6. SAVE RESULTS
    # ======================================

    base_name = os.path.splitext(image_name)[0]

    original_output = os.path.join(
        OUTPUT_FOLDER,
        "original_images",
        base_name + ".png"
    )

    mask_output = os.path.join(
        OUTPUT_FOLDER,
        "binary_masks",
        base_name + ".png"
    )

    overlay_output = os.path.join(
        OUTPUT_FOLDER,
        "overlays",
        base_name + ".png"
    )

    cv2.imwrite(
        original_output,
        image
    )

    cv2.imwrite(
        mask_output,
        binary_mask
    )

    cv2.imwrite(
        overlay_output,
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    # ======================================
    # 7. CALCULATE SEGMENTATION INFORMATION
    # ======================================

    wound_pixels = int(
        np.count_nonzero(binary_mask)
    )

    total_pixels = binary_mask.size

    wound_percentage = (
        wound_pixels / total_pixels
    ) * 100

    if wound_pixels == 0:
        empty_masks += 1

    report.append([
        image_name,
        mask_name,
        image.shape[1],
        image.shape[0],
        wound_pixels,
        round(wound_percentage, 2)
    ])

    processed += 1

    # Save first 3 samples for visualization
    if len(preview_items) < 3:

        preview_items.append((
            image_rgb.copy(),
            binary_mask.copy(),
            overlay.copy(),
            image_name
        ))

    # Progress
    if processed % 100 == 0:

        print("Processed:", processed)


# ==========================================
# 8. CREATE VISUALIZATION
# ==========================================

if preview_items:

    fig, axes = plt.subplots(
        len(preview_items),
        3,
        figsize=(15, 5 * len(preview_items))
    )

    if len(preview_items) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (image, mask, overlay, name) in enumerate(
        preview_items
    ):

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Original Image")

        axes[i, 1].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[i, 1].set_title("Binary Mask")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Wound Segmentation")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()

    preview_path = os.path.join(
        OUTPUT_FOLDER,
        "preview.png"
    )

    plt.savefig(
        preview_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

# ==========================================
# 9. SAVE CSV REPORT
# ==========================================

report_path = os.path.join(
    OUTPUT_FOLDER,
    "segmentation_report.csv"
)

with open(
    report_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Original Image",
        "Mask Image",
        "Width",
        "Height",
        "Wound Pixels",
        "Wound Percentage"
    ])

    writer.writerows(report)

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n===================================")
print("SEGMENTATION COMPLETED")
print("===================================")

print("Total Images:", len(images))
print("Total Masks:", len(masks))
print("Matched Pairs:", len(matched_files))
print("Processed Successfully:", processed)
print("Failed Images:", failed)
print("Unmatched Images:", len(unmatched_images))
print("Empty Masks:", empty_masks)

print("\nResults saved in:", OUTPUT_FOLDER)

print("Original Images: outputs/original_images")
print("Binary Masks: outputs/binary_masks")
print("Overlays: outputs/overlays")
print("Report: outputs/segmentation_report.csv")

print("\nALL TASKS FINISHED!")
