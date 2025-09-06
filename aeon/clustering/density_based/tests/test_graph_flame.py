"""Plot cluster labels with centers marked for Flame dataset."""

import matplotlib.pyplot as plt
import numpy as np

from aeon.clustering.density_based._density_peak import DensityPeakClusterer

# Flame dataset
test_data = np.array(
    [
        [1.85, 27.8],
        [1.35, 26.65],
        [1.4, 23.25],
        [0.85, 23.05],
        [0.5, 22.35],
        [0.65, 21.35],
        [1.1, 22.05],
        [1.35, 22.65],
        [1.95, 22.8],
        [2.4, 22.45],
        [1.8, 22],
        [2.5, 21.85],
        [2.95, 21.4],
        [1.9, 21.25],
        [1.35, 21.45],
        [1.35, 20.9],
        [1.25, 20.35],
        [1.75, 20.05],
        [2, 20.6],
        [2.5, 21],
        [1.7, 19.05],
        [2.4, 20.05],
        [3.05, 20.45],
        [3.7, 20.45],
        [3.45, 19.9],
        [2.95, 19.5],
        [2.4, 19.4],
        [2.4, 18.25],
        [2.85, 18.75],
        [3.25, 19.05],
        [3.95, 19.6],
        [2.7, 17.8],
        [3.45, 18.05],
        [3.8, 18.55],
        [4, 19.1],
        [4.45, 19.9],
        [4.65, 19.15],
        [4.85, 18.45],
        [4.3, 18.05],
        [3.35, 17.3],
        [3.7, 16.3],
        [4.4, 16.95],
        [4.25, 17.4],
        [4.8, 17.65],
        [5.25, 18.25],
        [5.75, 18.55],
        [5.3, 19.25],
        [6.05, 19.55],
        [6.5, 18.9],
        [6.05, 18.2],
        [5.6, 17.8],
        [5.45, 17.15],
        [5.05, 16.55],
        [4.55, 16.05],
        [4.95, 15.45],
        [5.85, 14.8],
        [5.6, 15.3],
        [5.65, 16],
        [5.95, 16.8],
        [6.25, 16.4],
        [6.1, 17.45],
        [6.6, 17.65],
        [6.65, 18.3],
        [7.3, 18.35],
        [7.85, 18.3],
        [7.15, 17.8],
        [7.6, 17.7],
        [6.7, 17.25],
        [7.3, 17.25],
        [6.7, 16.8],
        [7.3, 16.65],
        [6.75, 16.3],
        [7.4, 16.2],
        [6.55, 15.75],
        [7.35, 15.8],
        [6.8, 14.95],
        [7.45, 15.1],
        [6.85, 14.45],
        [7.6, 14.6],
        [8.55, 14.65],
        [8.2, 15.5],
        [7.9, 16.1],
        [8.05, 16.5],
        [7.8, 17],
        [8, 17.45],
        [8.4, 18.1],
        [8.65, 17.75],
        [8.9, 17.1],
        [8.4, 17.1],
        [8.65, 16.65],
        [8.45, 16.05],
        [8.85, 15.35],
        [9.6, 15.3],
        [9.15, 16],
        [10.2, 16],
        [9.5, 16.65],
        [10.75, 16.6],
        [10.45, 17.2],
        [9.85, 17.1],
        [9.4, 17.6],
        [10.15, 17.7],
        [9.85, 18.15],
        [9.05, 18.25],
        [9.3, 18.7],
        [9.15, 19.15],
        [8.5, 18.8],
        [11.65, 17.45],
        [11.1, 17.65],
        [10.4, 18.25],
        [10, 18.95],
        [11.95, 18.25],
        [11.25, 18.4],
        [10.6, 18.9],
        [11.15, 19],
        [11.9, 18.85],
        [12.6, 18.9],
        [11.8, 19.45],
        [11.05, 19.45],
        [10.3, 19.4],
        [9.9, 19.75],
        [10.45, 20],
        [13.05, 19.9],
        [12.5, 19.75],
        [11.9, 20.05],
        [11.2, 20.25],
        [10.85, 20.85],
        [11.4, 21.25],
        [11.7, 20.6],
        [12.3, 20.45],
        [12.95, 20.55],
        [12.55, 20.95],
        [12.05, 21.25],
        [11.75, 22.1],
        [12.25, 21.85],
        [12.8, 21.5],
        [13.55, 21],
        [13.6, 21.6],
        [12.95, 22],
        [12.5, 22.25],
        [12.2, 22.85],
        [12.7, 23.35],
        [13, 22.7],
        [13.55, 22.2],
        [14.05, 22.25],
        [14.2, 23.05],
        [14.1, 23.6],
        [13.5, 22.8],
        [13.35, 23.5],
        [13.3, 24],
        [7.3, 19.15],
        [7.95, 19.35],
        [7.7, 20.05],
        [6.75, 19.9],
        [5.25, 20.35],
        [6.15, 20.7],
        [7, 20.7],
        [7.6, 21.2],
        [8.55, 20.6],
        [9.35, 20.5],
        [8.3, 21.45],
        [7.9, 21.6],
        [7.15, 21.75],
        [6.7, 21.3],
        [5.2, 21.1],
        [6.2, 21.95],
        [6.75, 22.4],
        [6.15, 22.5],
        [5.65, 22.2],
        [4.65, 22.55],
        [4.1, 23.45],
        [5.35, 22.8],
        [7.4, 22.6],
        [7.75, 22.1],
        [8.5, 22.3],
        [9.3, 22],
        [9.7, 22.95],
        [8.8, 22.95],
        [8.05, 22.9],
        [7.6, 23.15],
        [6.85, 23],
        [6.2, 23.25],
        [5.7, 23.4],
        [5.1, 23.55],
        [4.55, 24.15],
        [5.5, 24],
        [6.1, 24.05],
        [6.5, 23.6],
        [6.75, 23.95],
        [7.3, 23.75],
        [8.3, 23.4],
        [8.9, 23.7],
        [9.55, 23.65],
        [10.35, 24.1],
        [7.95, 24.05],
        [3.95, 24.4],
        [3.75, 25.25],
        [3.9, 25.95],
        [4.55, 26.65],
        [5.25, 26.75],
        [6.5, 27.6],
        [7.45, 27.6],
        [8.35, 27.35],
        [9.25, 27.2],
        [9.95, 26.5],
        [10.55, 25.6],
        [9.9, 24.95],
        [9.2, 24.5],
        [8.55, 24.2],
        [8.8, 24.8],
        [9.2, 25.35],
        [9.55, 26.05],
        [9.05, 26.6],
        [8.8, 25.8],
        [8.15, 26.35],
        [8.05, 25.8],
        [8.35, 25.2],
        [7.9, 25.3],
        [8.05, 24.7],
        [7.3, 24.4],
        [7.55, 24.85],
        [6.85, 24.45],
        [6.25, 24.65],
        [5.55, 24.5],
        [4.65, 25.1],
        [5, 25.55],
        [5.55, 26.1],
        [5.55, 25.25],
        [6.2, 25.2],
        [6.8, 25.05],
        [7.4, 25.25],
        [6.65, 25.45],
        [6.15, 25.8],
        [6.5, 26.1],
        [6.6, 26.6],
        [7.7, 26.65],
        [7.5, 26.2],
        [7.5, 25.65],
        [7.05, 25.85],
        [6.9, 27.15],
        [6.15, 26.9],
    ]
)


def plot_cluster_labels_and_decision_graph():
    """Plot cluster labels with centers marked and decision graph sequentially."""
    # euclidean distance
    clusterer = DensityPeakClusterer(
        distance_metric="euclidean",
        density_threshold=8,
        distance_threshold=5,
        abnormal=False,
    )

    clusterer.fit(test_data)

    # Cluster labels visualization
    plt.figure(figsize=(12, 8))
    scatter1 = plt.scatter(
        test_data[:, 0],
        test_data[:, 1],
        c=clusterer.labels_,
        cmap="tab10",
        s=40,
        alpha=0.7,
    )

    # cluster centers
    center_points = test_data[clusterer.cluster_centers]
    plt.scatter(
        center_points[:, 0],
        center_points[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidth=2,
        label=f"Centers ({len(clusterer.cluster_centers)})",
    )

    plt.title(
        "Flame Dataset - Cluster Labels (Euclidean Distance)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("X coordinate", fontsize=12)
    plt.ylabel("Y coordinate", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter1, label="Cluster ID")
    plt.tight_layout()
    plt.show()

    # Decision graph (rho vs delta)
    plt.figure(figsize=(12, 8))
    scatter2 = plt.scatter(
        clusterer.rho,
        clusterer.delta,
        c=clusterer.labels_,
        cmap="tab10",
        s=40,
        alpha=0.7,
    )

    # cluster centers in decision graph
    center_rho = clusterer.rho[clusterer.cluster_centers]
    center_delta = clusterer.delta[clusterer.cluster_centers]
    plt.scatter(
        center_rho,
        center_delta,
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidth=2,
        label=f"Centers ({len(clusterer.cluster_centers)})",
    )

    # Add threshold lines to decision graph
    plt.axhline(
        y=5, color="red", linestyle="--", alpha=0.7, label="Distance Threshold = 5"
    )
    plt.axvline(
        x=8, color="orange", linestyle="--", alpha=0.7, label="Density Threshold = 8"
    )

    plt.title("Flame Dataset - Decision Graph (ρ vs δ)", fontsize=16, fontweight="bold")
    plt.xlabel("Local Density (ρ)", fontsize=12)
    plt.ylabel("Min Distance to Higher Density (δ)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter2, label="Cluster ID")
    plt.tight_layout()
    plt.show()

    # ============================================================
    # FLAME DATASET CLUSTERING RESULTS
    # ============================================================
    # Total data points: 240
    # Number of cluster centers: 2
    # Cluster centers indices: [68, 229]
    # Cutoff distance (dc): 1.0308
    # Unique cluster labels: [68, 229]
    #
    # Cluster size breakdown:
    # Cluster 68: 121 points (50.4%)
    # Cluster 229: 119 points (49.6%)
    #
    # Cluster centers in decision graph:
    # Center 1 (index 68): position=(7.30, 17.25), ρ=9.6218, δ=8.0006
    # Center 2 (index 229): position=(7.40, 25.25), ρ=9.6675, δ=14.7585
    #
    # Density (ρ) range: 0.2288 to 9.6675
    # Delta (δ) range: 0.4031 to 14.7585
    #
    # Threshold analysis:
    # Points with ρ ≥ 8: 30
    # Points with δ ≥ 5: 2
    # Points with both ρ ≥ 8 AND δ ≥ 5: 2

    return clusterer


if __name__ == "__main__":
    plot_cluster_labels_and_decision_graph()
