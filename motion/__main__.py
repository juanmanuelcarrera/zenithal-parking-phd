"""Run the five motion detectors over a sequence of frames and write a report."""
import argparse
from pathlib import Path

from . import (GMGMotionDetector, MOG2MotionDetector, MOGMotionDetector, MotionDetectionEvaluator,
               ThresholdDifferenceMotionDetector, WeightedAverageMotionDetector)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", help="folder with ordered frames")
    parser.add_argument("-o", "--output", default="runs/motion")
    args = parser.parse_args()

    frames = sorted(str(p) for p in Path(args.frames).iterdir() if p.suffix.lower() in {".jpg", ".png"})

    evaluator = MotionDetectionEvaluator(args.output)
    for detector in (MOG2MotionDetector(), MOGMotionDetector(), GMGMotionDetector(),
                     WeightedAverageMotionDetector(), ThresholdDifferenceMotionDetector()):
        evaluator.add_detector(detector)
    evaluator.load_test_images(frames)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
