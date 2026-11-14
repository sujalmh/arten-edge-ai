# Model exploration on the Pi 5

Every number here is measured on the Raspberry Pi 5 (8 GB, Bookworm, prebuilt
ExecuTorch 1.1.0 + torch 2.10), on the same input frame
(`data/samples/bengaluru_traffic.jpg`, 1040×780) at confidence 0.2 unless noted.
Counts are per frame at that threshold; no ground-truth labels exist for this
image, so the manual count is approximate.

## Manual reference

A 2×2 crop count of the input image gives roughly:

| Class | Manual estimate |
|---|---:|
| person | 30–40 |
| car | 45–55 |
| motorcycle | 25–30 |
| bus | 1 (plus distant buses) |
| truck | 3–4 |
| auto-rickshaw (not COCO) | 3 |

## Detection models

| Model | Mode | p50 | person | car | motorcycle | bus | truck | Drop-in? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n | full | 136.73 ms | 8 | 25 | 3 | 2 | 2 | yes |
| YOLO11n | 2×2 | 547.65 ms | 26 | 40 | 13 | 4 | 3 | yes |
| **YOLOv5s** | **full** | **149.74 ms** | **29** | **26** | **15** | 2 | 2 | yes |
| **YOLOv5s** | **2×2** | **601.73 ms** | **29** | **41** | **25** | 7* | 4 | yes |
| YOLOv8s | full | 233.81 ms | 17 | 27 | 14 | 1 | 2 | yes |
| YOLOv8s | 2×2 | 935.22 ms | 27 | 41 | 19 | 4 | 3 | yes |
| YOLOv9s | full | 508.62 ms | 17 | 25 | 10 | 1 | 3 | yes |
| YOLOv9s | 2×2 | 2036.9 ms | 25 | 33 | 16 | 5 | 2 | yes |
| YOLO12l | full | 1523.7 ms | 13 | 16 | 18 | 1 | 1 | yes |
| YOLO12l | 2×2 | 6102.9 ms | 23 | 20 | 24 | 2 | 4 | yes |

`*` Bus/truck over-count in tiled mode: a large vehicle spanning crops produces
partial boxes that the class-wise NMS does not merge. People, cars and
motorcycles are the reliable tiled classes.

**YOLOv5s is the standout.** It is faster than YOLOv8s full-frame (149.74 vs
233.81 ms), has higher person recall (29 vs 17) and higher two-wheeler recall
(15 vs 14) on this scene. Tiled, it reaches 29/41/25 at 601.73 ms, close to the
manual 30–40/45–55/25–30 and 1.5× faster than tiled YOLOv8s. Its COCO mAP is
lower (36.09 vs 43.86), so this is a domain result, not a general accuracy claim.

## Models that are not drop-in

| Model | Forward latency | Contract | Why it is excluded |
|---|---:|---|---|
| YOLOX-s | 232.29 ms | raw `(8400,85)` grid | needs grid/stride decode; a first decoder attempt over-counted, not validated |
| YOLOv8s-seg | 336.43 ms | flat 1,793,604 = 116×8400 detections + 32×160×160 masks | needs a mask-aware decoder; not validated |
| DETR-ResNet50 | 1661.4 ms at 800×800 | 100 queries: class logits + boxes + features | different input size and query decoder |
| PIDNet-s | 401.0 ms at 2048×1024 | 19-class Cityscapes mask | DDRNet gives 11 classes at 86 ms, 4.7× faster |

These were loaded, executed and their output contracts inspected. They are not
used in the product path because an unvalidated decoder would be an assumption,
not a verified result.

## Support models

| Model | Task | Latency | Status |
|---|---|---:|---|
| Lite-HRNet18 | top-down 17-keypoint pose | 142 ms/person | integrated |
| DDRNet23-slim | 11-class road mask | 86 ms/frame | integrated |
| YOLO11n-pose | full-frame pose | 234 ms (card) | tested; weak on the dense frame (one rider detected) |

Balanced support path on the 150-frame clip: YOLO11n 190.6 ms / 5.25 FPS,
YOLOv8s 291.1 ms / 3.43 FPS, YOLOv5s 210.8 ms / 4.74 FPS. Pose every 10 frames
(max 3), segmentation every 30.

## Manual checks

- **Input:** dense Bengaluru traffic with ~30–40 people, ~45–55 cars, ~25–30
  two-wheelers, one pink bus, several trucks, three auto-rickshaws.
- **YOLOv5s full:** 29 person boxes and 15 motorcycle boxes sit on real riders
  and pedestrians; car boxes align; no obvious duplicate person boxes.
- **YOLOv5s tiled:** 29/41/25 boxes align; the bus count (7) is visibly
  over-counted from partial boxes, so it is reported as unreliable.
- **YOLOv8s tiled:** 27/41/19, boxes align; same bus caveat.
- **Support frame:** HRNet skeletons align with head, shoulders, hips, knees and
  ankles; labels show `road`/`sidewalk` and `walking`/`standing`/`riding`.
- **Segmentation overlay:** road and sidewalk are broadly correct around the
  crossing; grass in the foreground is often labelled road because the model has
  no grass class.

## Recommendation

1. **Default for this domain:** YOLOv5s, full-frame for speed (6.7 FPS), 2×2
   tiled for accuracy (601.73 ms).
2. **Highest COCO mAP:** YOLOv8s, when general COCO accuracy matters more than
   domain recall or latency.
3. **Lowest cost:** YOLO11n, 136.73 ms / 7.31 FPS.
4. **Support detail:** Lite-HRNet18 + DDRNet, balanced cadence.
5. **Do not ship** YOLOX, YOLOv8s-seg, DETR or PIDNet without a validated
   decoder; their forward latencies are recorded but their counts are not
   comparable.

## Caveats

- No ground-truth labels for the image; the manual count is approximate.
- Counts depend on the confidence threshold; 0.2 was chosen from the earlier
  sweep, not tuned per model.
- Tiled bus/truck counts are unreliable.
- The Pi 5 numbers use the prebuilt multi-threaded runtime; Arm's cards are
  single-threaded and are not directly comparable.
