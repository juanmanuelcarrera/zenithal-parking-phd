PY ?= python

.PHONY: setup detectron2 rust proto test clean

setup:
	$(PY) -m pip install -e ".[detection,messaging,dev]"

detectron2:
	$(PY) -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

rust:
	cd core-rs && maturin develop --release

proto:
	protoc --python_out=messaging -I messaging messaging/parking_event.proto

data/ZenithalParking: data/ZenithalParking.zip
	unzip -q -o $< -d data

test:
	$(PY) -m pytest
	cd core-rs && cargo test

clean:
	rm -rf runs build *.egg-info core-rs/target messaging/parking_event_pb2.py
	find . -name __pycache__ -prune -exec rm -rf {} +
