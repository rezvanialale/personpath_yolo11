# YOLOv11 PersonPath22 Detection Project

This project evaluates YOLOv11 object detection models on the PersonPath22 video dataset.

## Project Goal

The goal is to run YOLOv11 person detection on PersonPath22 videos and evaluate detection performance using COCO-style metrics such as mAP, mAP@0.5, and mAP@0.75.

## Dataset

Dataset: PersonPath22  
Dataset source: Amazon Science PersonPath22 Tracking Dataset

The dataset is not included in this repository because the video files are large.  
Users should download the dataset separately and update the dataset path in `config.py`.

Expected dataset structure:

```text
tracking-dataset/
└── dataset/
    └── personpath22/
        ├── annotation/
        │   ├── anno_visible/
        │   └── anno_amodal/
        └── raw_data/
            └── videos/
                ├── uid_vid_00000.mp4
                ├── uid_vid_00001.mp4
                └── ...
