# Copyright (C) 2025 Ismail Pazarbasi
import numpy as np
import time
import matplotlib.pyplot as plt


class ProcessorConfig:
    def __init__(
        self,
        l1_cache_size_mb,
        l2_cache_size_mb,
        num_cores,
        simd_width_bits,
        l2_to_l1_bandwidth_gbps,
        l3_to_l2_bandwidth_gbps,
    ):
        self.l1_cache_size = l1_cache_size_mb * 1024 * 1024  # in bytes
        self.l2_cache_size = l2_cache_size_mb * 1024 * 1024  # in bytes
        self.num_cores = num_cores
        self.simd_width_bits = simd_width_bits
        self.l2_to_l1_bandwidth = (
            l2_to_l1_bandwidth_gbps * 1024 * 1024 * 1024
        )  # in bytes/s
        self.l3_to_l2_bandwidth = (
            l3_to_l2_bandwidth_gbps * 1024 * 1024 * 1024
        )  # in bytes/s

    def theoretical_peak_performance(self, clock_speed_ghz, dtype_bytes):
        """Calculates the theoretical peak performance in TOPS."""
        operations_per_lane_per_cycle = 1  # Assuming one op per lane per cycle
        total_operations_per_cycle = (
            operations_per_lane_per_cycle
            * self.num_cores
            * (self.simd_width_bits // (dtype_bytes * 8))
        )
        peak_ops_per_second = total_operations_per_cycle * clock_speed_ghz * 1e9
        return peak_ops_per_second / 1e12

    def calculate_flops(self, m, n, k):
        """Calculates the number of floating-point operations (FLOPs) for GEMM."""
        return (1 + 1) * m * n * k  # one mul + one add; no FMA

    # FIXME: Implement this in C.
    def run_tiled_gemm(self, a, b, tile_size, dtype):
        """Performs tiled GEMM and measures the execution time."""
        m, k = a.shape
        k_b, n = b.shape
        c = np.zeros((m, n), dtype=dtype)

        start_time = time.time()
        for i in range(0, m, tile_size):
            for j in range(0, n, tile_size):
                for l in range(0, k, tile_size):
                    a_tile = a[i : i + tile_size, l : l + tile_size]
                    b_tile = b[l : l + tile_size, j : j + tile_size]
                    c[i : i + tile_size, j : j + tile_size] += np.dot(
                        a_tile.astype(np.float32),
                        b_tile.astype(
                            # Using float32 for intermediate to handle different input types
                            np.float32
                        ),
                    ).astype(dtype)
        end_time = time.time()
        return end_time - start_time

    def estimate_l1_cache_misses(
        self, a_tile_size, b_tile_size, element_size, tile_size, cache_line_size=64
    ):
        """Estimates L1 cache misses considering cache line size."""
        return self._estimate_cache_misses(
            a_tile_size,
            b_tile_size,
            element_size,
            tile_size,
            cache_line_size,
            self.l1_cache_size,
        )

    def estimate_l2_cache_misses(
        self, a_tile_size, b_tile_size, element_size, tile_size, cache_line_size=64
    ):
        """Estimates L2 cache misses considering cache line size."""
        return self._estimate_cache_misses(
            a_tile_size,
            b_tile_size,
            element_size,
            tile_size,
            cache_line_size,
            self.l2_cache_size,
        )

    def _estimate_cache_misses(
        self,
        a_tile_size,
        b_tile_size,
        element_size,
        tile_size,
        cache_line_size,
        cache_level_size,
    ):
        """Estimates L1 cache misses considering cache line size."""

        def calculate_bytes_accessed(
            tile_size_rows, tile_size_cols, element_size, cache_line_size
        ):
            total_bytes = tile_size_rows * tile_size_cols * element_size
            return (
                (total_bytes + cache_line_size - 1) // cache_line_size * cache_line_size
            )

        a_bytes = calculate_bytes_accessed(
            a_tile_size, tile_size, element_size, cache_line_size
        )
        b_bytes = calculate_bytes_accessed(
            tile_size, b_tile_size, element_size, cache_line_size
        )
        c_bytes = calculate_bytes_accessed(
            a_tile_size, b_tile_size, element_size, cache_line_size
        )

        l1_footprint = a_bytes + b_bytes + c_bytes
        return l1_footprint > cache_level_size

    def estimate_l2_cache_misses(
        self, matrix_a_size, matrix_b_size, matrix_c_size, element_size
    ):
        """Estimates if the entire matrices fit in L2 cache. Simplistic model."""
        total_matrix_size = (
            matrix_a_size + matrix_b_size + matrix_c_size
        ) * element_size
        return total_matrix_size > self.l2_cache_size


def generate_roofline_data(
    processor, matrix_size, tile_sizes, clock_speed_ghz, dtype=np.int8
):
    """Generates data points for the roofline model."""
    roofline_data = []
    m, k = matrix_size
    _, n = matrix_size
    a = np.random.randint(
        np.iinfo(np.int8).min, np.iinfo(np.int8).max, size=(m, k), dtype=dtype
    )
    b = np.random.randint(
        np.iinfo(np.int8).min, np.iinfo(np.int8).max, size=(k, n), dtype=dtype
    )

    element_size = dtype().nbytes

    peak_performance = processor.theoretical_peak_performance(
        clock_speed_ghz, dtype().nbytes
    )

    for tile_size in tile_sizes:
        execution_time = processor.run_tiled_gemm(a, b, tile_size, dtype)
        flops = processor.calculate_flops(m, n, k)
        achieved_performance = flops / execution_time / 1e12  # in TOPS

        # Simplified model for arithmetic intensity (FLOPs/byte)
        bytes_loaded_per_tile = (
            2 * tile_size * tile_size * element_size
        )  # Loading A and B tiles
        flops_per_tile = 2 * tile_size * tile_size * tile_size
        arithmetic_intensity = (
            flops_per_tile / bytes_loaded_per_tile if bytes_loaded_per_tile > 0 else 0
        )

        l1_miss = processor.estimate_l1_cache_misses(
            tile_size, tile_size, element_size, tile_size
        )

        l2_miss = processor.estimate_l2_cache_misses(
            tile_size, tile_size, element_size, tile_size
        )

        memory_bandwidth_limited_performance_l1 = (
            processor.l2_to_l1_bandwidth
            / (bytes_loaded_per_tile / flops_per_tile)
            / 1e12
            if flops_per_tile > 0
            else 0
        )
        memory_bandwidth_limited_performance_l2 = (
            processor.l3_to_l2_bandwidth
            / (bytes_loaded_per_tile / flops_per_tile)
            / 1e12
            if flops_per_tile > 0
            else 0
        )

        roofline_data.append(
            {
                "tile_size": tile_size,
                "achieved_performance": achieved_performance,
                "arithmetic_intensity": arithmetic_intensity,
                "l1_miss": l1_miss,
                "l2_miss": l2_miss,
                "bandwidth_limit_l1": min(
                    peak_performance, memory_bandwidth_limited_performance_l1
                ),
                "bandwidth_limit_l2": min(
                    peak_performance, memory_bandwidth_limited_performance_l2
                ),
            }
        )
    return roofline_data


def plot_roofline(roofline_data, peak_performance):
    """Plots the roofline model with improved annotation placement."""
    intensities = [data["arithmetic_intensity"] for data in roofline_data]
    achieved_performances = [data["achieved_performance"] for data in roofline_data]
    l1_limits = [data["bandwidth_limit_l1"] for data in roofline_data]
    l2_limits = [data["bandwidth_limit_l2"] for data in roofline_data]

    plt.figure(figsize=(10, 6))
    plt.plot(
        intensities,
        achieved_performances,
        marker="o",
        linestyle="-",
        label="Achieved Performance (Various Tile Sizes)",
    )

    for x, y in zip(intensities, achieved_performances):
        label = f"{int(x)}"
        plt.annotate(
            label, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    plt.plot(intensities, l1_limits, linestyle="--", label="L1 Bandwidth Limit")
    plt.plot(intensities, l2_limits, linestyle="--", label="L2 Bandwidth Limit")
    plt.axhline(
        y=peak_performance,
        color="r",
        linestyle="-",
        label=f"Peak Performance ({peak_performance:.2f} TOPS)",
    )
    plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
    plt.ylabel("Performance (TOPS)")
    plt.title("Roofline Model for GEMM on SIMD Processor")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    plt.savefig('figure_1.svg', format='svg')
    plt.show()


if __name__ == "__main__":
    # Adjust this processor configuration.
    l1_size = 1  # MB
    l2_size = 2  # MB
    num_simd_cores = 4 * 4
    simd_lane_width_bits = 512  # bits
    l2_l1_bw = 20  # GB/s
    l3_l2_bw = 8  # GB/s
    clock = 2.0  # GHz

    # Matrix Size
    matrix_dimension = 1024  # (N x N)
    matrix_size = (matrix_dimension, matrix_dimension)

    # Tile Sizes to Test
    tile_sizes_to_test = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    # Create the SIMD processor object
    processor = ProcessorConfig(
        l1_size,
        l2_size,
        num_simd_cores,
        simd_lane_width_bits,
        l2_l1_bw,
        l3_l2_bw,
    )

    # Data type to use for GEMM
    dtype = np.int8
    # data_type = np.int16
    # data_type = np.float32

    # Generate roofline data
    roofline_data = generate_roofline_data(
        processor, matrix_size, tile_sizes_to_test, clock
    )
    peak_perf = processor.theoretical_peak_performance(clock, dtype().nbytes)

    # Print the results
    print(f"Theoretical Peak Performance: {peak_perf:.2f} TOPS")
    print("\nRoofline Data:")
    for data in roofline_data:
        print(f"  Tile Size: {data['tile_size']}")
        print(f"    Achieved Performance: {data['achieved_performance']:.2f} TOPS")
        print(
            f"    Arithmetic Intensity: {data['arithmetic_intensity']:.2f} FLOPs/Byte"
        )
        print(f"    L1 Cache Miss Estimate: {data['l1_miss']}")
        print(f"    L2 Cache Miss Estimate: {data['l2_miss']}")
        print(f"    L1 Bandwidth Limit: {data['bandwidth_limit_l1']:.2f} TOPS")
        print(f"    L2 Bandwidth Limit: {data['bandwidth_limit_l2']:.2f} TOPS")
        print("-" * 30)

    # Plot the roofline model
    plot_roofline(roofline_data, peak_perf)
