# Submission checklist

Use this before submitting the application (deadline **6 November 2026**) and
again before the live hackathon (14 November 2026).

---

## A. Five-minute pre-submit check

- [ ] Author is **Sujal M H**, solo participant, in `README.md`,
      `docs/report.md` and `docs/form-answers.md`.
- [ ] GitHub repository is **public** and the URL is inserted in the form.
- [ ] `README.md` renders correctly on GitHub: images load, links resolve, and
      the clone URL points to the public repository.
- [ ] `results/demo.mp4` plays in a browser (H.264, 21 s, 1280×720).
- [ ] `results/demo_pedestrians/contact_sheet.jpg` and the two annotated
      Bengaluru images are visible, with direction/colour labels.
- [ ] `results/demo_pedestrians/attribute_tracks.csv` and
      `results/demo_traffic/attribute_summary.json` are present.
- [ ] `make test` passes (29 tests).
- [ ] `make doctor` passes on the development host.
- [ ] `docs/report.pdf` is generated and opens; check page breaks and images.
- [ ] No secrets, API keys, personal phone numbers or private paths in the repo.
- [ ] `.venv/`, `__pycache__/`, raw downloads and scratch results are ignored.
- [ ] Model licence notice (`models/LICENSE-AGPL-3.0.txt`) and repo `LICENSE`
      are present.
- [ ] Sample-media attribution is present in `data/README.md`.
- [ ] The report's claims match the repository's numbers and labels.

## B. Mandatory deliverables map

| Requirement | Evidence in repo | Check |
|---|---|---|
| Process a 10–20 s clip or 10–20 images on ExecuTorch + Pi | `data/samples/pedestrians_15s.mp4`; `arten run`; Pi setup script | [ ] |
| Detect at least one category and show per-frame counts | `results/demo_pedestrians/counts.csv`; `summary.txt`; HUD | [ ] |
| Submit category + counts + two sample outputs | Contact sheet + traffic image + street image | [ ] |
| Report average inference time, model, throughput | `summary.json`; `benchmark_dev.json`; report §4.4 | [ ] |
| Explain society, help, one application (≤2 min) | `docs/pitch-2min.md`; report §1, §5 | [ ] |
| **Differentiator:** attributes alongside detections | `attribute_tracks.csv`; `attribute_summary.json`; README “Standout” | [ ] |

## C. Form-field checklist

The portal may ask for some or all of these. Drafts are in
[`form-answers.md`](form-answers.md).

- [ ] Author: **Sujal M H**
- [ ] Team size: **1 (solo participant)**
- [ ] Email and phone
- [ ] College/organisation and nomination (if a student)
- [ ] Problem statement selection: **Problem Statement 2**
- [ ] Project title: **Arten — Edge Object Detection, Counting & Attributes**
- [ ] 50-word elevator pitch
- [ ] 150-word summary
- [ ] 300-word solution description
- [ ] 500-word technical approach
- [ ] Innovation/differentiation
- [ ] Feasibility/execution plan
- [ ] Impact/measurable outcomes
- [ ] GitHub URL
- [ ] Demo video URL or file
- [ ] Declaration / eligibility confirmation

## D. Report quality checklist

- [ ] The first 100 words name a specific user and a specific decision, not
      “smart cities”.
- [ ] The mandatory deliverables are explicitly mapped.
- [ ] The model, runtime, size, latency, memory and accuracy are stated.
- [ ] The attribute layer is described as **derived** (image-plane direction,
      uncalibrated speed, heuristic colour), not as a trained model.
- [ ] Dev-host numbers are labelled as dev-host; Pi numbers are labelled as
      reference.
- [ ] The COCO/India gap and small-object failure are named.
- [ ] The privacy position is clear: no identities, no cloud.
- [ ] The 6-hour plan is realistic and has fallbacks.
- [ ] The report names Sujal M H as the sole author.
- [ ] The 2-minute script fits the time limit when read aloud.
- [ ] There is no claim of accuracy on data that was not measured.

## E. Two-minute pitch checklist

- [ ] Read aloud with a timer; finish under 2:00.
- [ ] Say “indicative” before the unique count.
- [ ] Say “reference” before the Pi 5 latency.
- [ ] Show the demo video or contact sheet at the right cue.
- [ ] Have the CSV and one annotated frame ready as backup.
- [ ] Rehearse the “if the demo fails” line.
- [ ] Prepare answers to the five likely questions in `pitch-2min.md`.
- [ ] Do not read the script word-for-word from a phone.

## F. Finals packing list (14 November)

**Hardware**
- [ ] Raspberry Pi 5 (8 GB) + official 27 W USB-C PD supply
- [ ] Active cooler / heatsink
- [ ] microSD card (32 GB+, A2) with a tested 64-bit Raspberry Pi OS image
- [ ] HDMI cable + micro-HDMI adapter
- [ ] USB keyboard and mouse (or a preconfigured SSH setup)
- [ ] Ethernet cable (venue Wi-Fi may be unreliable)
- [ ] USB stick with: repo, model, wheels/venv tarball, evidence bundles, demo video
- [ ] Phone hotspot as a backup (not needed for inference)

**Software / data**
- [ ] Fresh clone of the repo on the Pi
- [ ] `.venv` built and `doctor` passing
- [ ] Model checksum verified
- [ ] Organiser clip copied locally
- [ ] Output directory cleared for the live run
- [ ] Backup of `results/demo.mp4` and `results/demo_pedestrians/`

**Presentation**
- [ ] Printed `docs/report.pdf` (2 copies)
- [ ] Printed mandatory-deliverable table
- [ ] Laptop with the demo video and report
- [ ] Timer for the 2-minute pitch

## G. What not to do

- Do not add a dashboard or live camera on the day; it is out of scope and steals
  time from the mandatory evidence.
- Do not claim “real-time” without stating the FPS and the hardware.
- Do not claim accuracy on the organiser clip without ground truth.
- Do not hide the auto-rickshaw/motorcycle failures; judges will ask.
- Do not run the Pi on a phone charger or a weak power supply; undervoltage
  causes throttling and confusing latency.
- Do not leave the Pi in direct sun or on a soft surface during the demo.
- Do not rename the output files after the run; the report links to them.

## H. After the event

- [ ] Push the final Pi run and the organiser-clip evidence to GitHub.
- [ ] Replace the dev-host row in `README.md` and `docs/report.md` with the
      on-device numbers.
- [ ] Add a short “what I learned” note to the report.
- [ ] Keep the model checksum and licence notice with the final tag.
