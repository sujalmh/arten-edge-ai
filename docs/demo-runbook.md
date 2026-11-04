# Live hackathon runbook — 14 November 2026

A 6–8 hour plan that turns the existing PoC into a measured Pi result and a
2-minute pitch. The goal is **not** to add features; it is to run the mandatory
deliverables cleanly and leave a traceable evidence bundle.

---

## 0. Before you leave (night before)

- [ ] Flash a 64-bit Raspberry Pi OS image; boot it once; set username, Wi-Fi,
      SSH and locale.
- [ ] Clone the repo on the Pi, run `bash scripts/setup_pi.sh`, then
      `namma-drishti doctor`.
- [ ] Run the pedestrian demo once end-to-end; confirm `annotated.mp4` and
      `summary.txt`.
- [ ] Copy the model, repo, a wheel cache and the evidence bundles to a USB
      stick.
- [ ] Charge the laptop and phone; pack the power supply, HDMI adapter and
      Ethernet cable.
- [ ] Print two copies of `docs/proposal.md` (or the PDF) and the mandatory
      deliverable table.

## 1. Arrival and setup (0:00–0:45)

1. Claim a table near a power socket. Avoid direct sun and soft surfaces.
2. Connect the Pi to power, monitor, keyboard and Ethernet. Boot.
3. Verify the clock and free disk:
   ```bash
   date
   df -h
   free -h
   ```
4. Activate the venv and run the doctor:
   ```bash
   cd namma-drishti
   source .venv/bin/activate
   namma-drishti doctor
   ```
5. If the venv is missing, run `bash scripts/setup_pi.sh` (5–10 min). If the
   network is down, restore the prebuilt venv from the USB stick.
6. Run the 10-frame smoke test:
   ```bash
   namma-drishti run -i data/samples/pedestrians_15s.mp4 \
     -o results/smoke --max-frames 10 --annotate-every 5 --quiet
   ```
7. Confirm the summary says `checksum OK` and the model latency is plausible
   (Pi 5: roughly 120–150 ms; Pi 4: higher).

**If it fails:** check `namma-drishti doctor`, the model path, the Python
version and the ExecuTorch version. Do not debug for more than 10 minutes before
switching to the backup evidence and the dev host.

## 2. Organiser clip (0:45–1:30)

1. Copy the organiser clip into `data/organiser/`.
2. Inspect it:
   ```bash
   ffprobe -v error -show_entries \
     format=duration,size:stream=width,height,r_frame_rate,nb_frames \
     -of default=nw=1 data/organiser/clip.mp4
   ```
3. Run a 20-frame reconnaissance pass:
   ```bash
   namma-drishti run -i data/organiser/clip.mp4 \
     -o results/organiser_recon --max-frames 20 --annotate-every 5
   ```
4. Look at 3 annotated frames. Ask:
   - Are the target categories present and detected?
   - Is the confidence threshold too high (misses) or too low (false positives)?
   - Is the scene too dense/small for 640×640?
5. Record the clip properties and the chosen threshold in a notebook.

## 3. Threshold freeze (1:30–2:30)

1. Try `--conf 0.25`, `0.35`, `0.45` on the same 20 frames.
2. Compare counts and inspect the annotated frames. Pick one threshold and write
   it down. Do not tune per frame.
3. If the clip is 30 FPS, decide on `--stride 1` (quality) or `--stride 2`
   (speed). Report the stride in the summary.
4. Freeze: threshold, IoU, stride, classes, max frames.

**Rule:** the same settings must be used for the final run and the report. No
cherry-picking.

## 4. Final run and benchmark (2:30–3:30)

```bash
# Full clip, annotated video, evidence bundle
namma-drishti run -i data/organiser/clip.mp4 \
  -o results/organiser_final --save-video --annotate-every 10

# Model-only latency, 50 runs
namma-drishti benchmark --input data/organiser/frame.jpg \
  --runs 50 --warmup 10 --output results/benchmark_pi5.json
```

Check:
- `summary.txt` has counts, p50/p95, model-only and pipeline FPS.
- `counts.csv` has one row per processed frame.
- `annotated/` has at least two sample frames.
- `contact_sheet.jpg` is readable.
- `annotated.mp4` plays.
- `summary.json` says `checksum OK` and the correct device.

Copy the whole `results/organiser_final/` to the USB stick.

## 5. Report and assets (3:30–4:30)

1. Replace the dev-host row in `README.md` and `docs/report.md` with the Pi
   numbers from `benchmark_pi5.json` and `organiser_final/summary.json`.
2. Add the organiser-clip counts to the report's results table.
3. Regenerate the contact sheet if needed.
4. Update the demo video if time allows; otherwise use the committed 21-second
   video and the new contact sheet.
5. Write a one-paragraph “what the Pi measured” note with the exact command.

## 6. Pitch rehearsal (4:30–5:30)

1. Read `docs/pitch-2min.md` aloud with a timer three times.
2. Cut to under 2:00. Keep the hook, demo and impact.
3. Rehearse with the demo video playing silently.
4. Prepare the backup: `results/demo_pedestrians/contact_sheet.jpg` and
   `counts.csv` open on the laptop.
5. Prepare answers to the five questions in the pitch doc.

## 7. Buffer and submission (5:30–6:00)

- Push the final results and updated report to GitHub.
- Export the PDF if needed.
- Keep the Pi running with the final summary on screen.
- Do not start a new feature.

---

## Troubleshooting

### `namma-drishti: command not found`

```bash
source .venv/bin/activate
python -m namma_drishti doctor
```

If the package is missing: `pip install -e . --no-deps`.

### `executorch` import fails

Check the Python version (3.10–3.13) and reinstall:
```bash
pip install -r requirements-pi.txt
```
If the network is down, restore the prebuilt venv from the USB stick.

### `could not load ... with ExecuTorch`

- Confirm the model path and checksum.
- Confirm `executorch==1.1.0` on the Pi.
- Try `python -m namma_drishti info`.
- If it still fails, use the FP32 export fallback or the dev-host evidence.

### Video opens but no frames are read

The codec may be unsupported. Convert with ffmpeg:
```bash
ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p -an clip_h264.mp4
```

### No detections

- Lower `--conf` to 0.25.
- Confirm the requested classes are in the clip.
- Inspect a frame; the object may be too small or occluded.
- Check that preprocessing is not double-normalising (the pipeline handles it).

### Pi is slow / throttling

- Use the official 27 W supply.
- Add an active cooler; keep the Pi off soft surfaces.
- Use `--stride 2` or a shorter clip.
- Report the measured p95, not a best-case single run.

### Out of disk

```bash
rm -rf results/smoke results/organiser_recon
```
Keep the final bundle and the USB copy.

---

## Division of labour (3 people)

| Person | During setup | During run | During pitch |
|---|---|---|---|
| A | Pi OS, power, venv | Run CLI, benchmark, copy evidence | Demo operator |
| B | Repo, model, checksum | Inspect frames, tune threshold, fill report | Numbers and caveats |
| C | Laptop, USB, printouts | Build contact sheet/video, update README | Narrator |

If you are a team of one, do A and B first and rehearse C while the final run
is processing.
