# 1-Minute Live Demonstration Video Script (September 12 Submission)

**Project Title:** Real-Time Pothole Detection on Roads using Gradient-Based Edge Mapping  
**Format:** 60-Second Timed Screen Recording & Voiceover Demonstration  

---

## ⏱️ Second-by-Second Presentation Guide

```
[00:00 - 00:10] Problem Statement & Architectural Core
[00:10 - 00:25] Mathematical Pipeline Walkthrough (Unit 1 Concepts)
[00:25 - 00:45] Live System Demonstration (Dual Screen Feed)
[00:45 - 00:55] Quantitative Benchmark Results (284 FPS CPU)
[00:55 - 01:00] Conclusion & Delivery
```

---

### Segment 1: Introduction & Objective (0:00 - 0:10)
- **On Screen:** Title slide or IDE showing the project structure (`pothole_classical_detector.py`, `docs/`).
- **Spoken Audio:**
  > "Hello everyone. Today I'm demonstrating our real-time road pothole detection system, built entirely on classical mid-level computer vision and linear filtering principles from Unit 1, without relying on compute-heavy neural networks."

---

### Segment 2: Algorithmic Pipeline & Mathematical Theory (0:10 - 0:25)
- **On Screen:** Pipeline flowchart or source code snippet showing Gaussian -> Canny -> Morphology -> Contours.
- **Spoken Audio:**
  > "Our pipeline operates in five deterministic stages: first, we convert the raw BGR frame to single-channel intensity. Second, we apply a 5x5 Gaussian low-pass filter to attenuate high-frequency asphalt aggregate noise. Third, we compute spatial gradients using Sobel operators and Canny hysteresis thresholding to uncover structural boundaries. Fourth, morphological closing unifies fragmented edges. Finally, geometric area and aspect-ratio filtering isolate genuine potholes while rejecting linear lane markings."

---

### Segment 3: Live System Demonstration (0:25 - 0:45)
- **On Screen:** Run the live detector in terminal:
  ```bash
  ./.venv/bin/python pothole_classical_detector.py --source demo.mp4
  ```
  Show the side-by-side windows:
  1. **Left Window:** Live annotated road feed with red bounding boxes tracking potholes in real time.
  2. **Right Window:** Gradient Edge Map showing closed Canny contours.
- **Spoken Audio:**
  > "Here is the live demonstration running on our video stream. On the right, you can see the cleaned gradient edge map where broken aggregate noise is suppressed. On the left, our bounding boxes lock onto the road surface cavities as they approach, successfully filtering out both the solid shoulder lines and the dashed yellow lane dividers."

---

### Segment 4: Performance & Benchmarking (0:45 - 0:55)
- **On Screen:** Terminal output displaying the benchmark report:
  - Average Frame Latency: ~3.3 ms
  - Processing Rate: 284 FPS
  - Memory: < 35 MB
- **Spoken Audio:**
  > "In benchmark evaluations, our classical pipeline achieves an astonishing 284 frames per second on a commodity CPU, with an average frame latency of just 3.3 milliseconds. This provides massive headroom for automotive microcontrollers and embedded dashcam devices."

---

### Segment 5: Conclusion (0:55 - 1:00)
- **On Screen:** Concluding slide showing GitHub repository and Day-0 documentation suite.
- **Spoken Audio:**
  > "The Day-0 documentation suite, test datasets, and code are complete and validated. Thank you!"

---

## 🎬 How to Record the Demo on macOS

1. Open your terminal in the project directory:
   ```bash
   cd /Users/konthamsaisriharshith/Desktop/client
   ```
2. Start the detector:
   ```bash
   ./.venv/bin/python pothole_classical_detector.py --source demo.mp4
   ```
3. Press `Cmd + Shift + 5` to open macOS Screen Recorder, select the display area with the two OpenCV windows, hit **Record**, and read the script!
