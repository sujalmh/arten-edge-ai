# Two-minute pitch

**Goal:** make the judges remember one sentence — *count movement, not people* —
and believe the PoC is real, measured and deployable.

**Total:** ~300 words, about 2:00 at a relaxed pace. Rehearse with a timer.
**On screen:** `results/demo.mp4` playing silently, or the contact sheet plus the
terminal summary. Do not read slides.

---

## 0:00–0:12 — Hook

> Every school day, children cross roads that were designed before anyone counted
> them. The data to fix that is locked behind expensive cameras or cloud
> subscriptions.

*Hold up the Pi or point to it.*

> This is a sub-₹10,000 computer. It can count them offline.

## 0:12–0:30 — Problem

> India lost **1,68,491 people** on roads in 2022. Pedestrians were **one in
> five**; two-wheeler riders, **44.5%**. But a school committee doesn't need
> another surveillance system. It needs to know: how many children cross here at
> 3:30, and how much traffic is moving past them?

## 0:30–0:52 — Solution

> Namma Drishti is an offline edge counting kit. A 15-second clip goes in. A
> **4.7 MB** YOLO11n INT8 model runs through **ExecuTorch** on the Pi's CPU at
> **7.5 frames per second**. Out come per-frame counts for people, cars,
> motorcycles, buses and trucks. No cloud. No
> identities. Nothing leaves the device.

## 0:52–1:15 — Demo

*Play the clip or point to the contact sheet.*

> 150 frames, up to **9 people in one frame**, 18 indicative tracks. Every
> frame is in the CSV. Each track also carries image-plane direction,
> dwell time and relative speed; on vehicles, an optional coarse colour — nine
> white cars, four silver, two blue on the traffic image. This host takes
> **20 ms** per frame; Arm's Pi 5 reference is **134 ms**.

## 1:15–1:35 — Rigour

> We time every stage and report **p50 and p95**, not just an average. Attributes
> are derived: image-plane direction, uncalibrated speed, heuristic colour. The model card reports **38.9 mAP** on COCO, and we
> show where it fails: auto-rickshaws are not a COCO class, and small occluded
> motorcycles are missed in dense traffic.

## 1:35–1:55 — Impact

> A school records three 15-second clips — drop-off, dismissal, quiet hour —
> and runs them on the Pi, offline. If dismissal shows **40 children crossing with 60 vehicles**, that
> is a case for a crossing guard, a zebra crossing, or a no-parking zone at the
> gate. One Pi. No subscription.

## 1:55–2:00 — Close

> Count movement, not people. Namma Drishti.

*Stop. Don't fill the silence.*

---

## Three numbers to remember

| Number | Meaning |
|---|---|
| **4.73 MB** | The model that fits on the edge |
| **7.46 FPS / 134 ms** | Arm's verified Pi 5 reference |
| **1,68,491** | India's 2022 road deaths — the reason |

## If the live demo fails

Say this, then show the backup:

> The Pi is busy re-running the clip, so let me show the exact evidence bundle it
> produced — same command, same model, same CSV. The annotated frames and the
> per-frame counts are here on the USB.

Show `results/demo_pedestrians/contact_sheet.jpg` and `counts.csv`. Do not
apologise repeatedly; move to the impact line.

## Questions judges may ask

**“Is this just YOLO on a Pi?”**
> The value is the constraint-driven engineering: a verified INT8 ExecuTorch
> artifact, preprocessing matched to the export contract, per-stage timing, a
> one-command Pi setup, and an evidence bundle a non-technical committee can
> read. The model is a commodity; the deployment and the decision workflow are
> the work.

**“Why not use the cloud?”**
> The places that need this most have unreliable connectivity, and a school will
> not upload footage of children to a third-party server. Offline is both a
> cost decision and a trust decision.

**“How accurate is it?”**
> On COCO, the model card reports 38.9 mAP@0.5:0.95. On the organiser clip we
> report counts, not accuracy, because there is no ground truth. We show the
> failure cases instead of hiding them.

**“Can it detect auto-rickshaws?”**
> Not with COCO classes. That is a known gap and the next step is fine-tuning on
> an Indian-road dataset. For this PoC, we count the categories the approved
> model actually knows and label the limitation.

**“What is novel?”**
> The combination: a verified edge artifact, a full-pipeline measurement,
> on-device movement attributes, a privacy-preserving output, and a concrete
> civic workflow — at a cost a school can actually pay.

**“Are the attributes accurate?”**
> They are deliberately derived, not learned: direction is image-plane, speed is
> uncalibrated pixels/second, colour is a coarse heuristic. They are useful for
> comparing movement within one fixed-camera clip, not for enforcement.

**“Do you store faces or plates?”**
> No. The model has no face or plate head, and the pipeline never extracts
> identities. The annotated frames do contain the original pixels, so we treat
> them as sensitive evidence and blur them before publishing.

**“What would you do with two more weeks?”**
> Fine-tune on Indian street data, add tiled inference for small objects, and
> validate counts against manual ground truth on 20 clips. The architecture
> already supports swapping the model with `--model`.

## Delivery notes

- Say “**indicative**” before the unique count. It buys credibility.
- Say “**reference**” before the Pi 5 latency. Never imply the dev host is a Pi.
- Pause after “No identities.” Let the privacy point land.
- If you have 30 seconds cut, drop the rigour section and keep the hook, demo and
  impact.
