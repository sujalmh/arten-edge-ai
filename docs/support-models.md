# Support models: pose, surface and rider detail

Arten's detector answers *how many*. The optional support models answer *where,
on what surface, and in what posture* — without replacing the detector and
without a cloud call.

## Models

| Model | Task | Input | Output | Latency on Pi 5 | Licence |
|---|---|---|---|---:|---|
| Lite-HRNet18 INT8 | top-down 17-keypoint pose | 384×288 person crop, ImageNet norm | 17×96×72 heatmaps | **142 ms per person** | Apache-2.0 |
| DDRNet23-slim INT8 | 11-class road segmentation | 720×960, RGB − 128 | 11×90×120 logits | **86 ms per frame** | MIT |

Both `.pte` files are committed under `models/` with checksums; the runtime is
the same ExecuTorch stack as the detector.

## Fusion

- **Surface.** Segmentation runs every `--seg-every` frames. Each detection's
  foot point samples a 17×17 patch; road or sidewalk wins if it covers at least
  15% of the patch, otherwise the majority class. Output: `road`, `sidewalk`,
  `other`.
- **Rider.** A person box whose overlap with a motorcycle box is over 30% is
  tagged `rider`. Its surface is sampled below the motorcycle tyre, not at the
  rider's foot.
- **Posture.** The largest `--pose-max-persons` person boxes (default 3) get
  keypoints every `--pose-every` frames (default 10). Keypoints give
  `standing`, `walking`, `fallen`, `riding` or `unknown`. On video, tracker speed
  above 10 px/s promotes `standing` to `walking`.
- **Evidence.** Keypoints are drawn as a skeleton on annotated frames; the
  attribute CSV and summary gain `surface`, `posture` and `rider` fields.

## Measured performance (Pi 5, 150-frame clip)

| Configuration | Detector p50 | Pipeline mean | Throughput |
|---|---:|---:|---:|
| YOLO11n + pose/10 (max 3) + seg/30 | 136.73 ms | **190.6 ms** | **5.25 FPS** |
| YOLOv8s + pose/10 (max 3) + seg/30 | 233.81 ms | **291.1 ms** | **3.43 FPS** |
| YOLO11n, no support | 136.73 ms | 162 ms | 6.17 FPS |
| YOLOv8s, no support | 233.81 ms | — | 4.28 FPS |

Pose and segmentation are sampled, so their cost is amortised. On a support
frame the combined cost is about **454 ms** (three pose crops plus
segmentation); on the other nine frames it is zero.

For stills, run every frame:

```bash
arten run -i photo.jpg \
  --pose-model models/lite_hrnet18_int8_xnnpack.pte --pose-every 1 \
  --seg-model models/ddrnet23_slim_int8_xnnpack.pte --seg-every 1
```

Tiled YOLOv8s adds the accuracy gain: 935 ms detector + about 45 ms amortised
support, roughly 1 FPS on the Pi 5.

## Measured detail

Pedestrian clip, 150 frames, YOLOv8s at 0.2:

- 11 people peak, 31 person tracks, 1 car, 1 truck.
- Postures on sampled frames: walking 9, standing 2, unknown 2.
- Surfaces: road 480, other 309, sidewalk 99 detection-instances.
- Attribute tracks: 6 walking, 6 standing; skeletons align with people.

Dense Bengaluru image, YOLOv8s at 0.2 with every-frame support:

- 17 people, 27 cars, 14 motorcycles, 1 bus, 2 trucks.
- Riders tagged from motorcycle overlap; 2 sampled riders labelled `riding`.
- Surface attribution is weaker here: many detections sample `other` because
  the coarse segmentation sees vehicle bodies, not the road under them. The
  clip, with an open crossing, separates road and sidewalk cleanly.

## Limitations

- Surface and posture are **sampled and indicative**, not trained risk
  classifiers. Posture is a geometric heuristic (shoulder-hip angle, ankle
  spread, tracker speed), not a fall detector.
- Segmentation is low-resolution (90×120 logits upsampled) and has no grass
  class; grass is often labelled road. It is useful for open crossings, weaker
  in dense, occluded traffic.
- Pose runs on the **largest** people; small or distant riders may be skipped.
  `--pose-max-persons` caps cost.
- Support models add latency. The default cadence (pose every 10, segmentation
  every 30, three people) keeps the pipeline within 3.4–5.3 FPS on the Pi 5.
- Model licences: Lite-HRNet18 Apache-2.0, DDRNet23-slim MIT. Both are
  compatible with this AGPL-3.0 repository.
