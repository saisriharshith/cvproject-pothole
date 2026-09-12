# Real-Time Road Pothole Detection - Model & Pipeline Card

## 1. System Overview
- **Project Title:** Real-Time Pothole Detection on Roads using Gradient-Based Edge Mapping
- **System Classification:** Classical Computer Vision Pipeline (Unit 1: Linear Filters & Edge Detection) paired with an optional Deep Learning Object Detection verification layer (YOLOv12 / `best.pt`).
- **Primary Operational Objective:** Detect, bound, and highlight road surface structural anomalies (potholes) in real time from a forward-facing dashboard camera or vehicular video stream.
- **Hardware Target:** Standard commodity CPU (e.g., Apple Silicon M-series, Intel Core i5/i7, or embedded compute boards like Raspberry Pi 4/5 / Jetson Nano).
- **Latency & Throughput SLA:** 
  - Classical Mid-Level CV: **< 10 ms per frame** (> 100 FPS capability; benchmarked at ~3.3 ms / 284 FPS on Apple M-series).
  - Deep Learning Pipeline: **< 180 ms per frame** (> 5.5 FPS on CPU, > 30 FPS with GPU/Metal acceleration).

---

## 2. Mathematical & Algorithmic Specifications

```mermaid
graph LR
    A["Raw BGR Feed (1280x720)"] --> B["Grayscale Reduction (Y = 0.299R + 0.587G + 0.114B)"]
    B --> C["Gaussian Filter (5x5, σ=1.2) Noise Attenuation"]
    C --> D["Canny Edge Detection (Sobel Gradients Gx, Gy)"]
    D --> E["Morphological Closing (5x5 Elliptical Structuring Element)"]
    E --> F["Contour Extraction (RETR_EXTERNAL)"]
    F --> G["Geometric Heuristic Filter: Area (800-30000px) & Aspect Ratio (0.4-2.8)"]
    G --> H["Annotated Frame + Telemetry HUD"]
```

### 2.1 Classical Pipeline Stages
1. **Dimensionality Reduction:** Converts 3-channel BGR space into a single-channel intensity image:
   $$I(x, y) = 0.299 R + 0.587 G + 0.114 B$$
2. **High-Frequency Noise Suppression (Linear Shift-Invariant Filter):**
   Convolves $I(x, y)$ with a 2D Gaussian kernel $G(x, y)$ to smooth asphalt aggregate texture while preserving macroscopic structural discontinuities:
   $$G(x, y) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right), \quad \sigma = 1.2, \text{ Kernel Size} = 5\times 5$$
3. **Spatial Gradient Computation & Canny Mapping:**
   Computes directional derivatives using horizontal and vertical Sobel operators:
   $$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I_{\text{blur}}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I_{\text{blur}}$$
   Gradient Magnitude and Orientation:
   $$|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$
   Applies non-maximum suppression along the gradient normal and hysteresis thresholding with lower bound $\tau_{\text{low}} = 50$ and upper bound $\tau_{\text{high}} = 150$.
4. **Morphological Closing:**
   Applies dilation followed by erosion using an elliptical structuring element $B_{5\times 5}$ to bridge fragmented boundary segments resulting from non-uniform edge contrast:
   $$A \bullet B = (A \oplus B) \ominus B$$
5. **Structural Feature Discrimination (Geometric Heuristics):**
   - **Area Threshold:** $800 \le \text{Area}(C) \le 30,000 \text{ px}$ (eliminates micro-gravel noise and colossal horizon polygons).
   - **Aspect Ratio Constraint:** $0.4 \le \frac{\text{Width}}{\text{Height}} \le 2.8$ (rejects vertical lane markings where $\text{aspect ratio} < 0.25$ or horizontal road strips where $\text{aspect ratio} > 4.0$).

---

## 3. Operational Envelope & Boundary Constraints

### 3.1 In-Scope Conditions
- Standard asphalt and concrete road surfaces.
- Ambient daylight (overcast, midday sun, dawn/dusk with headlight assist).
- Forward-facing windshield mounting angle between $10^\circ$ and $25^\circ$ below horizontal.
- Vehicle speeds between 0 km/h and 90 km/h.

### 3.2 Out-of-Scope / Unsupported Conditions
- Severe weather: Heavy rain causing standing water reflection (specular reflection destroys intensity gradients).
- Unpaved dirt/gravel roads (excessive ambient texture variance exceeds Gaussian attenuation thresholds).
- Pitch black night driving without vehicular high-beam headlight illumination.
