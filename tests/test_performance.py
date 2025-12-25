#!/usr/bin/env python
"""Performance benchmarks for GaitTransformer inference.

Run with:
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_performance.py -v --benchmark-only

Or for quick timing without pytest-benchmark:
    CUDA_VISIBLE_DEVICES=0 python tests/test_performance.py
"""

import os
import time
from pathlib import Path

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import pytest

# Get the tests directory
TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"


@pytest.fixture(scope="module")
def model():
    """Load the GaitTransformer model once per test module."""
    from gait_transformer.gait_phase_transformer import load_default_model

    model = load_default_model()
    # Warm up the model with a dummy inference
    dummy_kp = np.random.randn(1, 90, 17, 3).astype(np.float32)
    dummy_h = np.array([1.7], dtype=np.float32)
    _ = model((dummy_kp, dummy_h), training=False)
    return model


@pytest.fixture(scope="module")
def test_data():
    """Load test keypoints data."""
    data = np.load(FIXTURES_DIR / "keypoints.npz")
    return {
        "small": data["keypoints_small"],
        "medium": data["keypoints"],
        "large": data["keypoints_large"],
        "height": float(data["height"][0]),
    }


class TestInferencePerformance:
    """Performance benchmarks for inference."""

    def test_inference_small(self, model, test_data, benchmark):
        """Benchmark inference on small dataset (500 frames)."""
        from gait_transformer.gait_phase_transformer import gait_phase_stride_inference

        keypoints = test_data["small"]
        height = test_data["height"]

        result = benchmark(gait_phase_stride_inference, keypoints, height, model, L=90)
        phases, stride = result
        assert phases.shape[0] == keypoints.shape[0]
        assert stride.shape[0] == keypoints.shape[0]

    def test_inference_medium(self, model, test_data, benchmark):
        """Benchmark inference on medium dataset (1055 frames)."""
        from gait_transformer.gait_phase_transformer import gait_phase_stride_inference

        keypoints = test_data["medium"]
        height = test_data["height"]

        result = benchmark(gait_phase_stride_inference, keypoints, height, model, L=90)
        phases, stride = result
        assert phases.shape[0] == keypoints.shape[0]
        assert stride.shape[0] == keypoints.shape[0]

    def test_inference_large(self, model, test_data, benchmark):
        """Benchmark inference on large dataset (3165 frames)."""
        from gait_transformer.gait_phase_transformer import gait_phase_stride_inference

        keypoints = test_data["large"]
        height = test_data["height"]

        result = benchmark(gait_phase_stride_inference, keypoints, height, model, L=90)
        phases, stride = result
        assert phases.shape[0] == keypoints.shape[0]
        assert stride.shape[0] == keypoints.shape[0]


class TestChunkingPerformance:
    """Benchmark the chunking/windowing functions."""

    def test_shift_generator(self, test_data, benchmark):
        """Benchmark the shift_generator function."""
        from gait_transformer.gait_phase_transformer import shift_generator

        keypoints = test_data["medium"]

        def run_shift():
            return list(shift_generator(keypoints, stride=1, L=90))

        result = benchmark(run_shift)
        assert len(result) == keypoints.shape[0] - 90 + 1

    def test_chunk_generator(self, test_data, benchmark):
        """Benchmark the chunk_generator function."""
        from gait_transformer.gait_phase_transformer import chunk_generator

        keypoints = test_data["medium"]

        def run_chunk():
            return list(chunk_generator(keypoints, stride=1, L=90, batch_size=128))

        result = benchmark(run_chunk)
        assert len(result) > 0


class TestXLACompilation:
    """Test XLA compilation and caching behavior."""

    def test_xla_caches_for_same_length(self, model, test_data):
        """Verify XLA results are cached for repeated input lengths."""
        from gait_transformer.gait_phase_transformer import gait_phase_stride_inference

        keypoints = test_data["medium"]
        height = test_data["height"]

        # First call - includes compilation
        start = time.time()
        _ = gait_phase_stride_inference(keypoints, height, model, L=90, use_xla=True)
        first_time = time.time() - start

        # Second call - should be cached
        start = time.time()
        _ = gait_phase_stride_inference(keypoints, height, model, L=90, use_xla=True)
        second_time = time.time() - start

        # Second call should be much faster (at least 10x)
        assert second_time < first_time / 5, f"Expected caching: first={first_time:.2f}s, second={second_time:.2f}s"

    def test_xla_padding_avoids_recompilation(self, model, test_data):
        """Verify pad_batches=True avoids recompilation for different lengths."""
        from gait_transformer.gait_phase_transformer import gait_phase_stride_inference, _compiled_predict_fns

        height = test_data["height"]

        # Clear cache to start fresh
        _compiled_predict_fns.clear()

        # First call - includes compilation for batch sizes 100 and 128
        kp1 = test_data["small"]  # 500 frames
        start = time.time()
        _ = gait_phase_stride_inference(kp1, height, model, L=90, use_xla=True, pad_batches=True)
        time1 = time.time() - start
        assert time1 > 1.0, f"Expected XLA compilation for first call: {time1:.2f}s"

        # Different length - should NOT recompile due to padding
        kp2 = test_data["medium"]  # 1055 frames
        start = time.time()
        _ = gait_phase_stride_inference(kp2, height, model, L=90, use_xla=True, pad_batches=True)
        time2 = time.time() - start
        assert time2 < 0.5, f"Expected cached (no recompile) with padding: {time2:.2f}s"

        # Another different length - should also be cached
        kp3 = test_data["large"]  # 3165 frames
        start = time.time()
        _ = gait_phase_stride_inference(kp3, height, model, L=90, use_xla=True, pad_batches=True)
        time3 = time.time() - start
        assert time3 < 0.5, f"Expected cached for large input: {time3:.2f}s"

    def test_xla_no_padding_recompiles(self, model, test_data):
        """Verify pad_batches=False causes recompilation for different lengths."""
        from gait_transformer.gait_phase_transformer import gait_phase_stride_inference, _compiled_predict_fns

        height = test_data["height"]

        # Clear cache
        _compiled_predict_fns.clear()

        # First call
        kp1 = test_data["small"]  # 500 frames
        start = time.time()
        _ = gait_phase_stride_inference(kp1, height, model, L=90, use_xla=True, pad_batches=False)
        time1 = time.time() - start

        # Different length without padding - should recompile
        kp2 = test_data["medium"]  # 1055 frames
        start = time.time()
        _ = gait_phase_stride_inference(kp2, height, model, L=90, use_xla=True, pad_batches=False)
        time2 = time.time() - start

        # Both should take significant time (compilation)
        assert time1 > 1.0, f"Expected compilation for first: {time1:.2f}s"
        assert time2 > 1.0, f"Expected recompilation without padding: {time2:.2f}s"

    def test_use_xla_false_no_compilation(self, model, test_data):
        """Verify use_xla=False avoids XLA compilation overhead."""
        from gait_transformer.gait_phase_transformer import gait_phase_stride_inference

        keypoints = test_data["medium"]
        height = test_data["height"]

        # With use_xla=False, first call should be fast (no XLA compilation)
        start = time.time()
        _ = gait_phase_stride_inference(keypoints, height, model, L=90, use_xla=False)
        first_time = time.time() - start

        # Should complete in under 2 seconds (no XLA compilation which takes 4-8s)
        assert first_time < 2.0, f"use_xla=False should not have XLA overhead: {first_time:.2f}s"


def run_quick_benchmark():
    """Run a quick benchmark without pytest-benchmark."""
    print("Loading model...")
    from gait_transformer.gait_phase_transformer import (
        load_default_model,
        gait_phase_stride_inference,
    )

    model = load_default_model()

    # Load test data
    data = np.load(FIXTURES_DIR / "keypoints.npz")
    keypoints = data["keypoints"]
    height = float(data["height"][0])

    print(f"\nBenchmarking with {keypoints.shape[0]} frames...")
    print("(First run includes XLA compilation, subsequent runs are faster)")

    # Run multiple times for timing
    times = []
    n_runs = 6
    for i in range(n_runs):
        start = time.perf_counter()
        phases, stride = gait_phase_stride_inference(keypoints, height, model, L=90)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        label = "(XLA compile)" if i == 0 else ""
        print(f"  Run {i+1}: {elapsed:.3f}s {label}")

    times = np.array(times)
    print(f"\nResults (excluding first run):")
    print(f"  Mean:   {times[1:].mean():.3f}s")
    print(f"  Std:    {times[1:].std():.3f}s")
    print(f"  Min:    {times[1:].min():.3f}s")
    print(f"  Max:    {times[1:].max():.3f}s")
    print(f"  Frames/sec: {keypoints.shape[0] / times[1:].mean():.1f}")

    return times[1:].mean()


if __name__ == "__main__":
    run_quick_benchmark()
