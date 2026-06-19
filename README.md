# Lightweight Anti-UAV Tracking System with Hybrid CPU-GPU Pipeline

A high-performance, modular Single Object Tracking (SOT) architecture designed specifically for autonomous drone detection and tracking under hardware-constrained environments (e.g., edge computing and airborne platforms). 

By implementing a **Heterogeneous CPU-GPU Pipeline**, this framework isolates traditional computer vision techniques, contour clustering, and mathematical motion modeling onto the CPU, while offloading batched deep-learning verification to the GPU. This decoupled load balancing yields robust target recovery and high-fidelity state estimation at near-real-time speeds on commodity hardware.

---

## 🚀 Key Architectural Features

1. **Heterogeneous Compute Pipeline (Load Balancing)**
   * **CPU Operations:** Background estimation using adaptive Gaussian Mixture Models (`MOG2`), morphological opening/closing kernels, cluster-based spatial merging, and Kalman matrix calculations.
   * **GPU Operations:** High-speed batched inference via `TinyPatchCNN` over candidate bounding box patches (`auto` mode dynamically targets CUDA execution if available) to filter out false alarms like birds, clouds, or sensor noise.

2. **Custom Kalman Box Filter (From-Scratch Mathematics)**
   * Implements a discrete 6D constant-velocity state vector $\mathbf{x} = [c_x, c_y, \dot{c_x}, \dot{c_y}, w, h]^T$ directly using NumPy, avoiding heavy tracking framework overhead.
   * Manages state covariance $\mathbf{P}$, dynamic process noise $\mathbf{Q}$, and measurement noise $\mathbf{R}$ matrices to reliably project trajectories during high-dynamics maneuvers.

3. **Occlusion-Aware Re-Acquisition Engine**
   * Combines spatial and appearance memory to robustly mitigate full or partial target occlusion (e.g., target passing behind structures/trees).
   * **Dynamic Expansion:** A localized Search ROI expanding proportionally with the tracking age/miss counter ($\text{roi\_scale} + \text{missed} \times 0.08$).
   * **Visual Memory:** Online adaptive HSV/grayscale color histogram profiling using correlation-based similarity matching (`cv2.HISTCMP_CORREL`) to ensure verified target re-locking.

4. **IoU-Constrained Synthetic Negative Sampling**
   * Dynamically extracts random background patches and clutter during the training data generation phase. By enforcing a rigorous Intersection-over-Union (IoU) constraint ($\le 0.05$) against the ground truth, it ensures pure background examples, which heavily regularizes the `TinyPatchCNN` against false positive alerts.

5. **Production-Grade MLOps & Benchmarking**
   * Built-in sequential metric collection to evaluate model metrics like absolute Recall, tracking continuity ratio, false alarm penalties per frame, and recovery rates across bulk datasets.

---

## 📊 Evaluation & Performance Metrics

The framework was comprehensively benchmarked across **67 high-fidelity drone tracking sequences** containing complex background variations, sudden scale changes, and prolonged occlusions:

| Metric | Benchmark Result | Operational Interpretation |
| :--- | :---: | :--- |
| **Processed Sequences** | `67` | Evaluated across diverse video streams |
| **Mean Frame Throughput** | **`20.99 FPS`** | Near-real-time execution using the hybrid pipeline |
| **Mean Recall** | **`79.56%`** | High-fidelity bounding box localization accuracy |
| **Tracked Frame Ratio** | **`97.65%`** | Sustained track retention across full video lifecycles |
| **Mean IoU on Hits** | **`0.6208`** | High-tightness tight-fitting box alignments |
| **False Positives Per Frame** | **`0.0193`** | Exceptionally low false alarm rate (~1.9% noise leakage) |
| **Mean Re-Acquisitions** | **`5.49`** | Successful target recovery events per sequence via visual memory |

---

## 📁 Repository Structure
```bash
├── core/
│   ├── data.py             # Highly optimized video cap stream wrapper and annotation parsing
│   ├── geometry.py         # Coordinate translations, clipping, and vectorized IoU calculations
│   ├── kalman.py           # Linear Kalman Filter implementation with 6D state representation
│   ├── metrics.py          # State evaluation metrics (Recall, False Positives, Tracked ratios)
│   ├── motion.py           # MOG2 background subtractor, morphological filters, & spatial cluster merging
│   ├── patch_classifier.py # Cropping pipeline and batched tensor loading for GPU processing
│   ├── patch_model.py      # Custom 3-layer Convolutional Neural Network with BatchNorm and SiLU
│   └── tracker.py          # Occlusion-Aware tracking coordinator (State machine)
├── outputs/                # Automated CSV output logs for benchmarks
├── weights/                # Serialized model weights (`tiny_patch_cnn.pt`)
├── run_tracker.py          # Single video execution script with live diagnostic overlay
├── run_benchmark.py        # Dataset benchmark suite computing full macro performance metrics
├── requirements.txt        # Project dependencies
└── train_patch_classifier.py # Classifier trainer with synthetic negative sampling

```

## 🔧 Installation & Setup
**Prerequisites**
* Python 3.8+
* OpenCV (opencv-python)
* PyTorch (with CUDA support for optimal GPU patch sorting)
* NumPy
```bash
# Clone the repository
git clone https://github.com/SelimOzn/uav.git
cd uav
# 2. Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
# 3. Install dependencies
pip install -r requirements.txt
```

## 💻 Usage Instructions
1. **Training the Patch Classifier**\
   To train the specialized `TinyPatchCNN` on a custom sequence partition with synthetic negative sampling:
   ```bash
   python train_patch_classifier.py --split-dir /path/to/dataset/train --modality visible --epochs 8 --batch-size 128 --lr 1e-3
   ```
2. **Running the Single Tracker (with Visualization)**\
   Run the tracking baseline on an individual sequence folder to inspect real-time Kalman adjustments, visual matches, and metrics:
   ```bash
   python run_tracker.py --sequence /path/to/dataset/val/sequence_01 --modality visible --patch-model weights/tiny_patch_cnn.pt 
   ```
3. **Evaluating Full Dataset Benchmarks**\
   To run quantitative evaluation over the validation/test dataset splits and dump results to a CSV log:
   ```bash
   python run_benchmark.py --split-dir /path/to/dataset/val --modality visible --csv outputs/lightweight_benchmark.csv
   ```
