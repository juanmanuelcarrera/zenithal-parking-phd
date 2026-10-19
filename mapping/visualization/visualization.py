"""Visualization utilities for parking space detection."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import random


def setup_matplotlib():
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8


def plot_image_comparison(images: Dict[str, np.ndarray], title: str = "Image Comparison",
                         figsize: Tuple[int, int] = (15, 10), cmap: str = 'gray') -> None:
    n_images = len(images)
    if n_images == 0:
        return

    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols

    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, fontweight='bold')

    for i, (img_title, img) in enumerate(images.items()):
        plt.subplot(rows, cols, i + 1)

        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        else:
            plt.imshow(img, cmap=cmap)

        plt.title(img_title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_processing_pipeline(stages: List[Tuple[str, np.ndarray]],
                           figsize: Tuple[int, int] = (20, 12)) -> None:
    n_stages = len(stages)
    if n_stages == 0:
        return

    cols = min(4, n_stages)
    rows = (n_stages + cols - 1) // cols

    plt.figure(figsize=figsize)
    plt.suptitle('Image Processing Pipeline', fontsize=18, fontweight='bold')

    for i, (stage_name, image) in enumerate(stages):
        plt.subplot(rows, cols, i + 1)

        if len(image.shape) == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        else:
            plt.imshow(image, cmap='gray')

        plt.title(f'{i+1}. {stage_name}', fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def draw_contours_on_image(image: np.ndarray, contours: List[np.ndarray],
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def draw_clustered_contours(image: np.ndarray, clustered_contours: List[List[np.ndarray]],
                           thickness: int = 2, show_centers: bool = True) -> np.ndarray:
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    colors = []
    for _ in clustered_contours:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)

    for cluster, color in zip(clustered_contours, colors):
        cv2.drawContours(result, cluster, -1, color, thickness)

        if show_centers:
            all_points = np.vstack(cluster)
            center = np.mean(all_points, axis=0).astype(int)
            cv2.circle(result, tuple(center), 5, color, -1)
            cv2.circle(result, tuple(center), 8, (255, 255, 255), 2)

    return result


def plot_contour_analysis(image: np.ndarray, contours: List[np.ndarray],
                         metrics: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Contour Analysis', fontsize=16, fontweight='bold')

    axes[0, 0].imshow(draw_contours_on_image(image, contours), cmap='gray')
    axes[0, 0].set_title(f'Detected Contours (Count: {metrics["count"]})')
    axes[0, 0].axis('off')

    if contours:
        areas = [cv2.contourArea(contour) for contour in contours]
        axes[0, 1].hist(areas, bins=20, alpha=0.7, color='blue')
        axes[0, 1].set_title('Contour Area Distribution')
        axes[0, 1].set_xlabel('Area (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

    if contours:
        perimeters = [cv2.arcLength(contour, True) for contour in contours]
        axes[1, 0].hist(perimeters, bins=20, alpha=0.7, color='green')
        axes[1, 0].set_title('Contour Perimeter Distribution')
        axes[1, 0].set_xlabel('Perimeter (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

    stats_text = f"""
    Total Contours: {metrics['count']}
    Total Area: {metrics['total_area']:.0f}
    Mean Area: {metrics['mean_area']:.1f} +/- {metrics['std_area']:.1f}
    Area Range: {metrics['min_area']:.0f} - {metrics['max_area']:.0f}
    Mean Perimeter: {metrics['mean_perimeter']:.1f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Statistics Summary')

    plt.tight_layout()
    plt.show()


def plot_clustering_analysis(image: np.ndarray, clustered_contours: List[List[np.ndarray]],
                           metrics: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Clustering Analysis', fontsize=16, fontweight='bold')

    axes[0, 0].imshow(draw_clustered_contours(image, clustered_contours))
    axes[0, 0].set_title(f'Clustered Contours ({metrics["num_clusters"]} clusters)')
    axes[0, 0].axis('off')

    if clustered_contours:
        axes[0, 1].bar(range(len(metrics['cluster_sizes'])), metrics['cluster_sizes'],
                      alpha=0.7, color='purple')
        axes[0, 1].set_title('Cluster Size Distribution')
        axes[0, 1].set_xlabel('Cluster Index')
        axes[0, 1].set_ylabel('Number of Contours')
        axes[0, 1].grid(True, alpha=0.3)

    if clustered_contours and len(clustered_contours) <= 6:
        n_clusters = len(clustered_contours)
        for i, cluster in enumerate(clustered_contours):
            if i < 6:
                cluster_img = draw_contours_on_image(image, cluster)
                row = 1 if i < 3 else 1
                col = i % 3 if i < 3 else (i % 3) + 1
                if i < 3:
                    axes[1, 0].imshow(cluster_img if i == 0 else cluster_img)
                    if i == 0:
                        axes[1, 0].set_title(f'Cluster 1 ({len(cluster)} contours)')
                        axes[1, 0].axis('off')

    stats_text = f"""
    Number of Clusters: {metrics['num_clusters']}
    Total Contours: {metrics['total_contours']}
    Mean Cluster Size: {metrics['mean_cluster_size']:.1f} +/- {metrics['std_cluster_size']:.1f}
    Size Range: {metrics['min_cluster_size']} - {metrics['max_cluster_size']}
    Cluster Sizes: {metrics['cluster_sizes']}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Clustering Statistics')

    plt.tight_layout()
    plt.show()


def plot_method_comparison(results: Dict[str, Dict[str, Any]],
                          metric: str = 'processing_time') -> None:
    valid_results = {name: res for name, res in results.items()
                    if 'error' not in res and name != 'comparison'}

    if not valid_results:
        print("No valid results to compare")
        return

    methods = list(valid_results.keys())
    values = []

    for method in methods:
        result = valid_results[method]
        if metric in result:
            values.append(result[metric])
        elif 'metadata' in result and metric in result['metadata']:
            values.append(result['metadata'][metric])
        else:
            values.append(0)

    if not values:
        print(f"Metric '{metric}' not found in results")
        return

    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods, values, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.title(f'Method Comparison: {metric.replace("_", " ").title()}', fontweight='bold')
    plt.xlabel('Methods')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45, ha='right')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_processing_report_visualization(report: Dict[str, Any],
                                        processed_image: Optional[np.ndarray] = None) -> None:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Processing Report: {report["method"]}', fontsize=16, fontweight='bold')

    method_info = f"""
    Method: {report['method']}
    Processing Time: {report['processing_time']:.4f} seconds
    Description: {report.get('metadata', {}).get('description', 'N/A')}
    """

    ax1 = plt.subplot(2, 3, 1)
    ax1.text(0.1, 0.7, method_info, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Method Information')

    if processed_image is not None:
        ax2 = plt.subplot(2, 3, 2)
        if len(processed_image.shape) == 3:
            ax2.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        else:
            ax2.imshow(processed_image, cmap='gray')
        ax2.set_title('Processed Image')
        ax2.axis('off')

    metrics_text = "Metrics:\n"

    if 'contour_metrics' in report:
        cm = report['contour_metrics']
        metrics_text += f"- Contours: {cm['count']}\n"
        metrics_text += f"- Mean Area: {cm['mean_area']:.1f}\n"
        metrics_text += f"- Total Area: {cm['total_area']:.0f}\n"

    if 'clustering_metrics' in report:
        clm = report['clustering_metrics']
        metrics_text += f"- Clusters: {clm['num_clusters']}\n"
        metrics_text += f"- Mean Cluster Size: {clm['mean_cluster_size']:.1f}\n"

    if 'edge_metrics' in report:
        em = report['edge_metrics']
        metrics_text += f"- Edge Density: {em['edge_density']:.4f}\n"
        metrics_text += f"- Edge Pixels: {em['edge_pixels']}\n"

    if 'image_quality' in report:
        iq = report['image_quality']
        metrics_text += f"- Contrast: {iq['contrast']:.1f}\n"
        metrics_text += f"- Sharpness: {iq['sharpness']:.1f}\n"
        metrics_text += f"- Entropy: {iq['entropy']:.2f}\n"

    ax3 = plt.subplot(2, 3, 3)
    ax3.text(0.1, 0.9, metrics_text, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Performance Metrics')

    plt.tight_layout()
    plt.show()


setup_matplotlib()
