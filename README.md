# Weakly Supervised Lesion Segmentation for Breast Ultrasound via Fine-grained Anchors and Segment Anything Model

This repository is the official implementation of "Weakly Supervised Lesion Segmentation for Breast Ultrasound via Fine-grained Anchors and Segment Anything Model". 

## Prerequisite
- Python 3.6, PyTorch 1.8.0, and more in requirements.txt
- CUDA 11.1
- 1 x  RTX 3090 GPUs

## Usage

### 1. Install python dependencies
```bash
python3 -m pip install -r requirements.txt
```

### 2. Generate Pixel-level Pseudo Annotations
- step1. Train classification network.
    ```python
    python s1_train_cam.py
- step2. Infer CAMs by using the trained CLs.Net.
    ```python 
    python s1_infer_cam.py  
- step3. Refer to the [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to derive class-specific SAM masks.
 
- step4. Selecting high-confidence SAM masks using a filtering formula.
    ```python 
    python s2_filter_formula.py
- step5. Optimize CAM using filtered SAM mask.
    ```python 
    python s2_refine_sam.py
### 3. Train Lesion Segmentation network
- step1. Train classification network using the Pixel-level Pseudo Annotations.
    ```python
    python s3_train_seg.py
- step2. Infer segmentation results by using the trained Seg.Net.
    ```python 
    python s3_infer_seg.py
