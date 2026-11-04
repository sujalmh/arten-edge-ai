PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: help setup test demo benchmark info doctor pdf demo-video clean

help:
	@echo "Namma Drishti"
	@echo "  make setup       install on Raspberry Pi (or a dev host with requirements-pi.txt)"
	@echo "  make test        run unit tests"
	@echo "  make demo        process the committed sample clip and images"
	@echo "  make benchmark   model-only latency on this device"
	@echo "  make demo-video  build results/demo.mp4 from the pedestrian run"
	@echo "  make info        show model/runtime information"
	@echo "  make doctor      check the environment"
	@echo "  make pdf         render docs/proposal.html to docs/proposal.pdf"
	@echo "  make clean       remove caches (keeps results)"

setup:
	bash scripts/setup_pi.sh

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m namma_drishti run -i data/samples/pedestrians_15s.mp4 \
		-o results/demo_pedestrians --save-video --annotate-every 10
	$(PY) -m namma_drishti run -i data/samples/bengaluru_traffic.jpg \
		-o results/demo_traffic --annotate-every 1 --no-track
	$(PY) -m namma_drishti run -i data/samples/bengaluru_street.jpg \
		-o results/demo_street --annotate-every 1 --no-track

benchmark:
	$(PY) -m namma_drishti benchmark --input data/samples/bengaluru_traffic.jpg \
		--runs 50 --warmup 10 --output results/benchmark_dev.json

demo-video:
	$(PY) scripts/make_demo_video.py

info:
	$(PY) -m namma_drishti info

doctor:
	$(PY) -m namma_drishti doctor

pdf:
	bash scripts/make_pdf.sh

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
