# Sample media

These files are for development and for the submitted evidence bundle. At the
finals, the organiser-provided clip/images replace them; the CLI is agnostic to
resolution, codec and frame rate.

## `pedestrians_15s.mp4`

- **What:** 15-second trim of the OpenCV `vtest.avi` pedestrian sequence.
- **Properties:** 640×480, 10 FPS, 150 frames, H.264.
- **Source:** <https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi>
- **Provenance:** a long-standing public OpenCV sample clip, widely used for
  optical-flow and pedestrian-detection tutorials.
- **Why:** it gives a clean, low-resolution pedestrian scene with people entering
  and leaving frame — exactly the counting behaviour the PoC needs to show.

## `bengaluru_traffic.jpg`

- **What:** elevated view of dense Bengaluru traffic.
- **Source:** Wikimedia Commons, [TRAFFIC VIEW IN BANGALORE.jpg](https://commons.wikimedia.org/wiki/File:TRAFFIC_VIEW_IN_BANGALORE.jpg)
- **Author:** Divya Chinnasamy
- **Licence:** CC0 1.0 (public-domain dedication)
- **Why:** a local, dense, real-world stress test. It exposes the small-object
  and motorcycle-recall limitations discussed in the report.

## `bengaluru_street.jpg`

- **What:** street-level scene with a car, an auto-rickshaw, a motorcycle and
  pedestrians.
- **Source:** Wikimedia Commons, [Traffic in Bangalore.JPG](https://commons.wikimedia.org/wiki/File:Traffic_in_Bangalore.JPG)
- **Author:** Chaitanya Lala
- **Licence:** public domain
- **Why:** a close-range scene where the detector's person/vehicle counts are
  easy to inspect, and where the missing auto-rickshaw class is obvious.

## Re-fetching

```bash
bash scripts/fetch_samples.sh
```

The script downloads the OpenCV clip through jsDelivr, trims it to 15 seconds
with ffmpeg, and fetches the two Wikimedia images. The committed files are the
exact inputs used for the results bundles.

## Attribution in outputs

Annotated images and the demo video are derivative works of these samples. If you
publish them, keep the attribution above and the licence notices in this file.
