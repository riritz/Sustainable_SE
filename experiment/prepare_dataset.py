import os
import json
import shutil
import random
from pathlib import Path

def reduce_dataset_with_annotations( src_images, src_ann, dst_images, dst_ann, n=100, seed=42):
    random.seed(seed)

    with open(src_ann) as f:
        coco_dataset = json.load(f)

    # Sample N images
    sampled_images = random.sample(coco_dataset["images"], min(n, len(coco_dataset["images"])))
    sampled_ids = {img["id"] for img in sampled_images}

    # Filter annotations to match sampled images
    sampled_annotations = [a for a in coco_dataset["annotations"] if a["image_id"] in sampled_ids]

    # Save new annotation file
    Path(dst_images).mkdir(parents=True, exist_ok=True)
    new_coco_dataset = {
        "info": coco_dataset.get("info", {}),
        "licenses": coco_dataset.get("licenses", []),
        "categories": coco_dataset["categories"],
        "images": sampled_images,
        "annotations": sampled_annotations
    }
    with open(dst_ann, "w") as f:
        json.dump(new_coco_dataset, f)

    # Copy images
    for img in sampled_images:
        src = os.path.join(src_images, img["file_name"])
        dst = os.path.join(dst_images, img["file_name"])
        if os.path.exists(src):
            shutil.copy(src, dst)

    print(f"[DATASET] {len(sampled_images)} images, {len(sampled_annotations)} annotations saved")

def check_annotations(images_dir, ann_file, name):
    print(f"\n[CHECK] {name}")

    with open(ann_file) as f:
        coco_dataset = json.load(f)

    ann_filenames  = {img["file_name"] for img in coco_dataset["images"]}
    disk_filenames = set(os.listdir(images_dir))

    missing_on_disk = ann_filenames - disk_filenames
    missing_in_ann  = disk_filenames - ann_filenames

    print(f"  Images in annotation:  {len(ann_filenames)}")
    print(f"  Images on disk:        {len(disk_filenames)}")
    print(f"  Missing on disk:       {len(missing_on_disk)}")
    print(f"  Missing in annotation: {len(missing_in_ann)}")

    if not missing_on_disk and not missing_in_ann:
        print("  OK — annotations and images match perfectly")
    else:
        if missing_on_disk:
            print(f"  WARNING: {list(missing_on_disk)[:5]} ...")
        if missing_in_ann:
            print(f"  WARNING: {list(missing_in_ann)[:5]} ...")

if __name__ == "__main__":
    # Warmup dataset - 500 images
    reduce_dataset_with_annotations(
        src_images="dataset/val2017",
        src_ann="dataset/annotations_trainval2017/annotations/instances_val2017.json",
        dst_images="dataset/val2017_warmup",
        dst_ann="dataset/annotations_trainval2017/annotations/instances_val2017_warmup.json",
        n=500,
        seed=42
    )

    # Energy measurement dataset - 1500 images
    reduce_dataset_with_annotations(
        src_images="dataset/val2017",
        src_ann="dataset/annotations_trainval2017/annotations/instances_val2017.json",
        dst_images="dataset/val2017_energy",
        dst_ann="dataset/annotations_trainval2017/annotations/instances_val2017_energy.json",
        n=1500,
        seed=99  # different seed so it's a different set of images
    )