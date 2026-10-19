"""Evaluator for comparing multiple motion detection algorithms."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

from .base import BaseMotionDetector
from .metrics import MotionDetectionMetrics


class MotionDetectionEvaluator:
    """Comprehensive evaluator for motion detection algorithms."""

    def __init__(self, output_dir: str = "runs/motion"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.detectors: List[BaseMotionDetector] = []
        self.metrics: Dict[str, MotionDetectionMetrics] = {}
        self.test_images: List[Tuple[np.ndarray, str]] = []

    def add_detector(self, detector: BaseMotionDetector) -> None:
        self.detectors.append(detector)
        self.metrics[detector.name] = MotionDetectionMetrics(detector.name)

    def load_test_images(self, image_paths: List[str]) -> None:
        self.test_images = []
        for path in image_paths:
            image = cv2.imread(path)
            if image is not None:
                self.test_images.append((image, Path(path).name))
            else:
                print(f"Warning: Could not load image {path}")

    def run_evaluation(self) -> Dict[str, Any]:
        if len(self.test_images) < 2:
            raise ValueError("Need at least 2 test images for motion detection")

        results = {}

        for detector in self.detectors:
            print(f"Evaluating {detector.name}...")
            detector_results = self._evaluate_single_detector(detector)
            results[detector.name] = detector_results

        self._generate_comparison_report(results)

        return results

    def _evaluate_single_detector(self, detector: BaseMotionDetector) -> Dict[str, Any]:
        metrics = self.metrics[detector.name]
        detection_results = []

        detector.reset_metrics()

        for i in range(len(self.test_images) - 1):
            current_image, current_name = self.test_images[i]
            next_image, next_name = self.test_images[i + 1]

            metrics.start_measurement()

            if i == 0:
                detector.initialize(current_image)
                has_motion, n_contours, mask, _ = detector.process_frame(next_image)
            else:
                has_motion, n_contours, mask, _ = detector.process_frame(next_image)

            result = metrics.record_result(has_motion, n_contours, mask)
            detection_results.append({
                'frame_pair': f"{current_name} -> {next_name}",
                'has_motion': has_motion,
                'n_contours': n_contours,
                'processing_time': result.processing_time
            })

        performance = metrics.get_performance_summary()

        return {
            'performance': performance,
            'detections': detection_results,
            'detector_metrics': detector.get_metrics_summary()
        }

    def _generate_comparison_report(self, results: Dict[str, Any]) -> None:
        performance_data = []
        for detector_name, detector_results in results.items():
            perf = detector_results['performance']
            performance_data.append({
                'Detector': detector_name,
                'Avg Processing Time (ms)': perf['avg_processing_time'] * 1000,
                'Avg FPS': perf['avg_fps'],
                'Avg CPU (%)': perf['avg_cpu_percent'],
                'Max Memory (MB)': perf['max_memory_mb'],
                'Motion Detection Rate': perf['motion_detection_rate'],
                'Avg Contours': perf['avg_contours_per_frame']
            })

        csv_path = self.output_dir / 'performance_comparison.csv'
        with open(csv_path, 'w') as f:
            headers = list(performance_data[0].keys())
            f.write(','.join(headers) + '\n')

            for row in performance_data:
                values = [str(row[header]) for header in headers]
                f.write(','.join(values) + '\n')

        self._create_performance_plots(performance_data, results)

        self._create_detailed_report(performance_data, results)

    def _create_performance_plots(self, performance_data: List[Dict], results: Dict[str, Any]) -> None:
        detectors = [row['Detector'] for row in performance_data]
        proc_times = [row['Avg Processing Time (ms)'] for row in performance_data]
        fps_values = [row['Avg FPS'] for row in performance_data]
        cpu_values = [row['Avg CPU (%)'] for row in performance_data]
        memory_values = [row['Max Memory (MB)'] for row in performance_data]
        detection_rates = [row['Motion Detection Rate'] for row in performance_data]
        contour_values = [row['Avg Contours'] for row in performance_data]

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        plt.figure(figsize=(10, 6))
        bars = plt.bar(detectors, proc_times, color=colors[:len(detectors)])
        plt.title('Mean Processing Time', fontsize=14, fontweight='bold')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
        for bar, value in zip(bars, proc_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(proc_times)*0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'chart_processing_time.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(detectors, fps_values, color=colors[:len(detectors)])
        plt.title('Frames per Second', fontsize=14, fontweight='bold')
        plt.ylabel('FPS')
        plt.xticks(rotation=45)
        for bar, value in zip(bars, fps_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fps_values)*0.01,
                   f'{value:.1f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'chart_fps.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(detectors, cpu_values, color=colors[:len(detectors)])
        plt.title('Mean CPU Usage', fontsize=14, fontweight='bold')
        plt.ylabel('CPU (%)')
        plt.xticks(rotation=45)
        for bar, value in zip(bars, cpu_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cpu_values)*0.01,
                   f'{value:.1f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'chart_cpu_usage.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(detectors, memory_values, color=colors[:len(detectors)])
        plt.title('Peak Memory', fontsize=14, fontweight='bold')
        plt.ylabel('Memory (MB)')
        plt.xticks(rotation=45)
        for bar, value in zip(bars, memory_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory_values)*0.01,
                   f'{value:.1f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'chart_memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(detectors, detection_rates, color=colors[:len(detectors)])
        plt.title('Motion Detection Rate', fontsize=14, fontweight='bold')
        plt.ylabel('Rate (0-1)')
        plt.xticks(rotation=45)
        for bar, value in zip(bars, detection_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(detection_rates)*0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'chart_detection_rate.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(detectors, contour_values, color=colors[:len(detectors)])
        plt.title('Mean Contours per Frame', fontsize=14, fontweight='bold')
        plt.ylabel('Contours')
        plt.xticks(rotation=45)
        for bar, value in zip(bars, contour_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(contour_values)*0.01,
                   f'{value:.1f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'chart_contours.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Motion Detection Benchmark', fontsize=16, fontweight='bold')

        ax = axes[0, 0]
        bars = ax.bar(detectors, proc_times, color=colors[:len(detectors)])
        ax.set_title('Mean Processing Time')
        ax.set_ylabel('Time (ms)')
        ax.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, proc_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(proc_times)*0.01,
                   f'{value:.2f}', ha='center', va='bottom')

        ax = axes[0, 1]
        bars = ax.bar(detectors, fps_values, color=colors[:len(detectors)])
        ax.set_title('Frames per Second')
        ax.set_ylabel('FPS')
        ax.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, fps_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fps_values)*0.01,
                   f'{value:.1f}', ha='center', va='bottom')

        ax = axes[1, 0]
        bars = ax.bar(detectors, cpu_values, color=colors[:len(detectors)])
        ax.set_title('Mean CPU Usage')
        ax.set_ylabel('CPU (%)')
        ax.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, cpu_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cpu_values)*0.01,
                   f'{value:.1f}%', ha='center', va='bottom')

        ax = axes[1, 1]
        bars = ax.bar(detectors, memory_values, color=colors[:len(detectors)])
        ax.set_title('Peak Memory')
        ax.set_ylabel('Memory (MB)')
        ax.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, memory_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory_values)*0.01,
                   f'{value:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'chart_performance_combined.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Individual charts saved:")
        print(f"  - {self.output_dir / 'chart_processing_time.png'}")
        print(f"  - {self.output_dir / 'chart_fps.png'}")
        print(f"  - {self.output_dir / 'chart_cpu_usage.png'}")
        print(f"  - {self.output_dir / 'chart_memory_usage.png'}")
        print(f"  - {self.output_dir / 'chart_detection_rate.png'}")
        print(f"  - {self.output_dir / 'chart_contours.png'}")
        print(f"  - {self.output_dir / 'chart_performance_combined.png'}")

        ax = axes[0, 0]
        bars = ax.bar(detectors, proc_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title('Mean Processing Time')
        ax.set_ylabel('Time (ms)')
        ax.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, proc_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(proc_times)*0.01,
                   f'{value:.2f}', ha='center', va='bottom')

        ax = axes[0, 1]
        bars = ax.bar(detectors, fps_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title('Frames per Second')
        ax.set_ylabel('FPS')
        ax.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, fps_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fps_values)*0.01,
                   f'{value:.1f}', ha='center', va='bottom')

        ax = axes[1, 0]
        bars = ax.bar(detectors, cpu_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title('Mean CPU Usage')
        ax.set_ylabel('CPU (%)')
        ax.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, cpu_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cpu_values)*0.01,
                   f'{value:.1f}%', ha='center', va='bottom')

        ax = axes[1, 1]
        bars = ax.bar(detectors, memory_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title('Peak Memory')
        ax.set_ylabel('Memory (MB)')
        ax.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, memory_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory_values)*0.01,
                   f'{value:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison_4plots.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Performance comparison plots saved to {self.output_dir / 'performance_comparison_4plots.png'}")

        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Performance plots saved to {self.output_dir / 'performance_comparison.png'}")

    def _create_detailed_report(self, performance_data: List[Dict], results: Dict[str, Any]) -> None:
        report_path = self.output_dir / 'detailed_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MOTION DETECTION BENCHMARK\n")
            f.write("=" * 80 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 30 + "\n")

            headers = ['Detector', 'Time (ms)', 'FPS', 'CPU (%)', 'Memory (MB)', 'Detection', 'Contours']
            f.write(f"{'Detector':<20} {'Time(ms)':<12} {'FPS':<8} {'CPU(%)':<8} {'Mem(MB)':<10} {'Detection':<10} {'Contours':<10}\n")
            f.write("-" * 80 + "\n")

            for row in performance_data:
                f.write(f"{row['Detector']:<20} ")
                f.write(f"{row['Avg Processing Time (ms)']:<12.2f} ")
                f.write(f"{row['Avg FPS']:<8.1f} ")
                f.write(f"{row['Avg CPU (%)']:<8.1f} ")
                f.write(f"{row['Max Memory (MB)']:<10.1f} ")
                f.write(f"{row['Motion Detection Rate']:<10.2f} ")
                f.write(f"{row['Avg Contours']:<10.1f}\n")

            f.write("\n\n")

            for detector_name, detector_results in results.items():
                f.write(f"DETAIL - {detector_name.upper()}\n")
                f.write("-" * 40 + "\n")

                perf = detector_results['performance']
                f.write(f"Processing time: {perf['avg_processing_time']*1000:.2f} +/- {perf['std_processing_time']*1000:.2f} ms\n")
                f.write(f"Rango de tiempo: {perf['min_processing_time']*1000:.2f} - {perf['max_processing_time']*1000:.2f} ms\n")
                f.write(f"FPS promedio: {perf['avg_fps']:.1f}\n")
                f.write(f"CPU: {perf['avg_cpu_percent']:.1f}% (max: {perf['max_cpu_percent']:.1f}%)\n")
                f.write(f"Memory: {perf['avg_memory_mb']:.1f} MB (max: {perf['max_memory_mb']:.1f} MB)\n")
                f.write(f"Motion detected: {perf['motion_detection_rate']*100:.1f}% of frames\n")
                f.write(f"Contours per frame: {perf['avg_contours_per_frame']:.1f} (max: {perf['max_contours_per_frame']})\n")
                f.write("\n")

        print(f"Detailed report saved to {report_path}")
