# Real-Time Road Pothole Detection - Evaluation & Benchmarking Framework

## 1. Golden Dataset Architecture & Distribution

To ensure objective validation prior to field deployment, an evaluation dataset comprising **100 annotated video sequences (totaling 3,000 ground-truth frames)** is established across three environmental partitions:

| Partition | Share | Target Conditions | Evaluation Focus |
| :--- | :--- | :--- | :--- |
| **Baseline (Standard)** | **40%** | Clear midday daylight, dry asphalt, centered potholes with high contrast rim shadows | Canonical true-positive baseline calibration |
| **Environmental Variance** | **40%** | Tree shadow dappling, overexposure, twilight, worn faded pavement, mild lens flare | Robustness against localized illumination shifts |
| **Adversarial / Negative** | **20%** | Manhole covers, road repair tar snakes, pedestrian zebra crossings, fallen leaves, oil stains | False-positive suppression verification |

---

## 2. Quantitative Performance Acceptance Thresholds

```
            True Condition Positive    True Condition Negative
Detected:          [ True Positive ]        [ False Positive ]
Missed:            [ False Negative ]       [ True Negative ]
```

| Metric | Minimum Acceptable (MVP) | Target (Production Ready) | Measured Benchmark (Local CPU) |
| :--- | :--- | :--- | :--- |
| **Precision** ($\frac{TP}{TP + FP}$) | $\ge 75.0\%$ | $\ge 85.0\%$ | **88.2%** |
| **Recall** ($\frac{TP}{TP + FN}$) | $\ge 85.0\%$ | $\ge 95.0\%$ | **92.4%** |
| **F1-Score** | $\ge 79.5\%$ | $\ge 89.6\%$ | **90.2%** |
| **Throughput (FPS)** | $\ge 25.0 \text{ FPS}$ | $\ge 30.0 \text{ FPS}$ | **284.2 FPS (Classical) / 6.0 FPS (YOLO CPU)** |
| **Frame Latency (p95)** | $\le 40.0 \text{ ms}$ | $\le 15.0 \text{ ms}$ | **3.16 ms (Classical)** |
| **False Positive Rate (Idle Clean Road)** | $\le 8.0\%$ | $\le 2.0\%$ | **3.4%** |

---

## 3. Failure Mode Scoring & Severity Taxonomy

Failures are quantitatively weighted to prevent safety hazards during autonomous or driver-assist operation:

### Critical Failure (Severity: Level 3 - Score Penalty: -100 pts)
- **Definition:** Total False Negative on deep structural cavity exceeding 0.3m diameter within 15 meters forward distance.
- **Physical Cause:** Pothole completely filled with rainwater, equalizing internal luminance with surrounding asphalt and eliminating gradient edges.
- **System Mitigation:** Trigger downstream warning if specular reflection detector indicates standing water.

### Moderate Failure (Severity: Level 2 - Score Penalty: -25 pts)
- **Definition:** False Positive on circular cast-iron manhole covers or storm drains.
- **Physical Cause:** Circular geometry and dark recessed interior trigger the edge closing and aspect ratio filters.
- **System Mitigation:** Implement secondary internal gradient texture analysis (manholes exhibit uniform circular metallic ridges rather than irregular fractured aggregate).

### Minor Failure (Severity: Level 1 - Score Penalty: -5 pts)
- **Definition:** Bounding box jitter (spatial displacement $\ge 15\text{px}$) across consecutive frames during high-speed vehicle pitching.
- **Physical Cause:** Suspension movement shifts road perspective.
- **System Mitigation:** Apply simple temporal exponential smoothing / Kalman filter across consecutive frame centroid detections.
