# Real-Time Road Pothole Detection - Data Strategy & Environmental Boundary Spec

## 1. Ingestion Protocol & Sensor Constraints

| Parameter | Specification | Hardware Rationale |
| :--- | :--- | :--- |
| **Sensor Type** | Standard CMOS Optical Dashboard Camera or Integrated Webcam | Cost-effective deployment without expensive LiDAR/radar |
| **Color Model & Encoding** | 8-bit standard BGR (RGB matrix in memory) | Native OpenCV ingestion standard |
| **Resolution Range** | $1280 \times 720 \text{ (720p)}$ native; downsampled to $640 \times 360$ for ultra-low latency if needed | Balancing spatial frequency fidelity against convolution cost |
| **Frame Capture Rate** | Minimum 30 FPS progressive scan | Prevents severe motion blur at cruising velocities ($\le 60 \text{ km/h}$) |
| **Field of View (FoV)** | $60^\circ$ to $90^\circ$ horizontal FoV | Ensures sufficient road surface coverage while minimizing fisheye barrel distortion |

---

## 2. Preprocessing & Conditioning Strategy

1. **Dimensionality Reduction:**
   Raw frames undergo immediate conversion to single-channel intensity, reducing memory throughput by $66.7\%$ ($3 \text{ bytes/px} \to 1 \text{ byte/px}$).
2. **Shift-Invariant Noise Conditioning:**
   Raw sensor noise and high-frequency asphalt micro-textures are attenuated via a $5 \times 5$ Gaussian kernel with standard deviation $\sigma = 1.2$. This isolates macroscopic intensity gradients associated with physical road fractures while eliminating false edge triggers from individual aggregate stones.
3. **Dynamic Contrast Adaptation (Degraded Lighting Mitigation):**
   When mean scene luminance $\mu_L < 40$, optional Contrast Limited Adaptive Histogram Equalization (CLAHE) is engaged across localized $8 \times 8$ grid tiles to recover low-contrast perimeter gradients.

---

## 3. Invariant Testing & Hyperparameter Adaptation

### Invariant Test 1: Severe Shadow Transitions (Tree Canopy / Overpass)
- **Observed Anomaly:** Harsh shadow boundaries project linear edges across the road, generating false Canny edges.
- **Mitigation Strategy:** Dynamic hysteresis adaptation. If average frame edge density exceeds $15\%$ of total frame pixels, raise lower Canny threshold $\tau_{\text{low}}$ from 50 to 90 and upper threshold $\tau_{\text{high}}$ from 150 to 220, requiring steeper spatial luminance gradients.

### Invariant Test 2: Partial Occlusion & Vehicle Preceding
- **Observed Anomaly:** Leading vehicles cast dark under-carriage shadows mimicking pothole cavities.
- **Mitigation Strategy:** Region of Interest (ROI) dynamic masking. Restrict contour search to the lower $60\%$ trapezoidal road polygon (ignoring the horizon and direct vehicle bumpers).

### Invariant Test 3: Water-Filled Cavities (Specular Surface)
- **Observed Anomaly:** Standing water creates mirror reflections, eliminating depth shadows.
- **Mitigation Strategy:** Failover to Deep Learning verification model (YOLOv12 / `best.pt`) or multi-frame temporal flow disparity.
