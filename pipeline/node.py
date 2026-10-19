"""
Processing node: event-driven sentinel/expert loop.

motion -> YOLO confirms vehicles -> wait N_s still frames -> Mask R-CNN
-> occupancy with hysteresis -> publish state changes.
A watchdog forces an expert pass every `watchdog_s` seconds.
"""
import argparse
import json
import time

import cv2

from detection.maskrcnn import MaskRCNNSegmenter
from detection.yolo import YoloSegmenter
from motion import ThresholdDifferenceMotionDetector
from occupancy import OccupancyTracker


class Node:
    def __init__(self, node_id, spots, sentinel, expert, publisher=None,
                 stable_frames: int = 15, watchdog_s: float = 300):
        self.node_id = node_id
        self.motion = ThresholdDifferenceMotionDetector()
        self.sentinel = sentinel
        self.expert = expert
        self.tracker = OccupancyTracker(spots, k=3)
        self.publisher = publisher
        self.stable_frames = stable_frames
        self.watchdog_s = watchdog_s

        self.pending = False
        self.still = 0
        self.last_expert = 0.0

    def step(self, frame):
        moving, _, _, _ = self.motion.process_frame(frame)

        if moving:
            self.still = 0
            if not self.pending and self.sentinel.has_vehicles(frame):
                self.pending = True
            return

        self.still += 1
        watchdog = time.time() - self.last_expert > self.watchdog_s
        if (self.pending and self.still >= self.stable_frames) or watchdog:
            self.run_expert(frame)

    def run_expert(self, frame):
        vehicles = [d.polygon for d in self.expert(frame)]
        changed = self.tracker.update(vehicles)
        self.pending = False
        self.last_expert = time.time()

        for spot_id, state in changed.items():
            print(f"{spot_id}: {state.value} ({self.tracker.ratio[spot_id]:.0%})")
            if self.publisher:
                from messaging.events import build_event
                event = build_event(self.node_id, int(spot_id.split("_")[-1]), state, self.tracker.ratio[spot_id])
                self.publisher.publish(event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="camera index or video file")
    parser.add_argument("--spots", required=True, help="output of python -m mapping")
    parser.add_argument("--maskrcnn-weights")
    parser.add_argument("--yolo-weights", default="yolo11n-seg.pt")
    parser.add_argument("--node-id", default="node_A")
    parser.add_argument("--mqtt-host")
    args = parser.parse_args()

    spots = {s["id"]: s["vertices"] for s in json.load(open(args.spots))["parking_spaces"]}

    publisher = None
    if args.mqtt_host:
        from messaging.publisher import EventPublisher
        publisher = EventPublisher(args.node_id, args.mqtt_host)

    node = Node(args.node_id, spots, YoloSegmenter(args.yolo_weights), MaskRCNNSegmenter(args.maskrcnn_weights),
                publisher)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        node.step(frame)
    cap.release()


if __name__ == "__main__":
    main()
