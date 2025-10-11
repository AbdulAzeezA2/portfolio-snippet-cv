# YOLOv5-PyTorch

![Example](https://github.com/AbdulAzeezA2/portfolio-snippet-cv/blob/main/YOLOv5-PyTorch/Infer/192.jpg)

A clean, minimal **PyTorch implementation of YOLOv5** designed for easy understanding and experimentation.  
This project demonstrates object detection on the **Rock–Paper–Scissors dataset** and serves as a learning-oriented version of YOLOv5.

---

## 🚀 Features

- **Pure Python implementation** — runs directly with PyTorch ≥ 1.4, no C/C++ build required  
- **Lightweight and modular** — easy to read and extend  
- **Compatible** with YOLOv5 architecture from [Ultralytics](https://github.com/ultralytics/yolov5)  
- **TorchVision-style structure** for better integration and flexibility  
- **Supports VOC and COCO datasets**, as well as custom datasets in COCO format  

---

## ⚙️ Requirements

- **Windows** or **Linux**
- **Python ≥ 3.6**
- **[PyTorch ≥ 1.6.0](https://pytorch.org/)**
- **matplotlib** – for image visualization
- **[pycocotools](https://github.com/cocodataset/cocoapi)** – for COCO dataset evaluation  
  - ⚠️ Windows users: install from [this fork](https://github.com/philferriere/cocoapi)
- *(Optional)* **[NVIDIA DALI](https://developer.download.nvidia.cn/compute/redist/nvidia-dali-cuda100/)** – for faster data loading (Linux only)

> **Note:** There are known print-related issues with `pycocotools` on Windows. See [Issue #356](https://github.com/cocodataset/cocoapi/issues/356).

---

## 🧠 Dataset

This repository supports **PASCAL VOC** and **COCO** datasets.  
You can train on your own dataset by either:
- Writing a compatible dataset loader, or  
- Converting your dataset into **COCO-style JSON format**.

**Example Datasets:**
- [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [MS COCO 2017](http://cocodataset.org/)
- **Dataset Used for Training and Inference:**  
  [Rock-Paper-Scissors (Roboflow)](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw/dataset/11)

---

## 🧩 Model Architecture

The model combines:
- **Darknet** as the backbone for feature extraction  
- **PANet (Path Aggregation Network)** for feature fusion and object detection  

You can view the architecture flowchart by opening  
`images/YOLOv5.drawio` with [draw.io](https://app.diagrams.net/).

---

## 🏋️ Training

Example command for training on COCO dataset (1 GPU):

```bash
python -m torch.distributed.run --nproc_per_node=1 --use_env train.py \
--use-cuda --dali --mosaic \
--epochs 190 --data-dir "./data/coco2017" \
--ckpt-path "yolov5s_coco.pth"
