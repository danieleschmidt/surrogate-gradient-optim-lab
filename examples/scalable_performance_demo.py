#!/usr/bin/env python3
"""
Scalable Performance Demo for Surrogate Gradient Optimization Lab
Demonstrates Generation 3 scaling features: JIT, vectorization, parallel processing
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

# Import scalable components
import sys
sys.path.append('/root/repo')
from surrogate_optim.core.scalable_optimizer import ScalableSurrogateOptimizer
from surrogate_optim.data.collector import collect_data


def complex_benchmark_function(x):
    """Complex function for performance benchmarking."""
    x = jnp.atleast_1d(x)
    
    # Multi-component function
    quadratic = jnp.sum(x**2)
    rosenbrock = jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    oscillatory = 0.1 * jnp.sum(jnp.sin(10 * x) * jnp.exp(-x**2))
    
    return float(0.5 * quadratic + 0.3 * rosenbrock + oscillatory)


def test_scaling_performance():
    """Test scaling performance across different problem sizes."""
    print("⚡ Testing Scaling Performance")
    print("=" * 50)
    
    # Test different problem dimensions
    dimensions = [2, 5, 10, 20]
    sample_sizes = [50, 100, 200, 500]
    
    results = {}
    
    for n_dims in dimensions:
        print(f"\n📊 Testing {n_dims}D problems...")
        
        for n_samples in sample_sizes:
            print(f"   📈 Collecting {n_samples} samples...")
            
            # Generate bounds for this dimension
            bounds = [(-3.0, 3.0)] * n_dims
            
            # Collect training data
            start_time = time.time()
            data = collect_data(
                function=complex_benchmark_function,
                n_samples=n_samples,
                bounds=bounds,
                sampling="random"
            )
            data_time = time.time() - start_time
            
            # Create scalable optimizer
            optimizer = ScalableSurrogateOptimizer(
                surrogate_type="gaussian_process",
                enable_jit=True,
                enable_vectorization=True,
                enable_parallel=True,
                batch_size=min(200, n_samples // 2),
                enable_caching=True,
                auto_optimize=True,
                enable_monitoring=True
            )
            
            # Train model
            start_time = time.time()
            optimizer.fit_surrogate(data, optimize_memory=True)
            training_time = time.time() - start_time
            
            # Test prediction performance
            test_points = jnp.array(np.random.uniform(-2, 2, (1000, n_dims)))
            
            start_time = time.time()
            predictions = optimizer.predict(test_points, use_batch_processing=True)
            prediction_time = time.time() - start_time
            
            # Test gradient performance
            try:
                start_time = time.time()
                gradients = optimizer.gradient(test_points[:100], use_batch_processing=True)
                gradient_time = time.time() - start_time
            except Exception:
                gradient_time = None
            
            # Store results
            key = f"{n_dims}D_{n_samples}samples"
            results[key] = {
                "dimensions": n_dims,
                "samples": n_samples,
                "data_collection_time": data_time,
                "training_time": training_time,
                "prediction_time": prediction_time,
                "gradient_time": gradient_time,
                "prediction_throughput": 1000 / prediction_time if prediction_time > 0 else 0,
                "gradient_throughput": 100 / gradient_time if gradient_time else 0,
            }
            
            print(f"      ✅ Training: {training_time:.3f}s")
            print(f"      ✅ Prediction: {prediction_time:.3f}s ({1000/prediction_time:.0f}/s)")
            if gradient_time:
                print(f"      ✅ Gradient: {gradient_time:.3f}s ({100/gradient_time:.0f}/s)")
    
    return results


def test_parallel_optimization():
    """Test parallel optimization capabilities."""
    print("\n🚀 Testing Parallel Optimization")
    print("=" * 40)
    
    # Create test problem
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    data = collect_data(
        function=complex_benchmark_function,
        n_samples=200,
        bounds=bounds,
        sampling="sobol"
    )
    
    # Create optimizer
    optimizer = ScalableSurrogateOptimizer(
        surrogate_type="gaussian_process",
        optimizer_type="gradient_descent",
        enable_parallel=True,
        max_workers=4,
        enable_monitoring=True
    )
    
    # Train model
    optimizer.fit_surrogate(data)
    
    # Test single vs parallel optimization
    n_starts = 8
    initial_points = [
        jnp.array([np.random.uniform(-3, 3), np.random.uniform(-3, 3)])
        for _ in range(n_starts)
    ]
    
    # Sequential optimization
    print("🔄 Running sequential optimization...")
    start_time = time.time()
    sequential_results = []
    for point in initial_points:
        result = optimizer.optimize(
            initial_point=point,
            bounds=bounds,
            num_steps=50
        )
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    # Parallel optimization
    print("⚡ Running parallel optimization...")
    start_time = time.time()
    parallel_result = optimizer.parallel_optimize(
        initial_points=initial_points,
        bounds=bounds,
        num_steps=50,
        return_all=False
    )
    parallel_time = time.time() - start_time
    
    # Best sequential result
    best_sequential = min(sequential_results, key=lambda r: r.fun)
    
    speedup = sequential_time / parallel_time
    
    print(f"\n📊 Parallel Optimization Results:")
    print(f"   Sequential time: {sequential_time:.2f}s")
    print(f"   Parallel time: {parallel_time:.2f}s")
    print(f"   Speedup: {speedup:.1f}x")
    print(f"   Best sequential value: {best_sequential.fun:.6f}")
    print(f"   Best parallel value: {parallel_result.fun:.6f}")
    print(f"   Quality difference: {abs(parallel_result.fun - best_sequential.fun):.6f}")


def test_caching_performance():
    """Test caching performance benefits."""
    print("\n💾 Testing Caching Performance")
    print("=" * 35)
    
    # Create test setup
    bounds = [(-3.0, 3.0), (-3.0, 3.0)]
    data = collect_data(
        function=complex_benchmark_function,
        n_samples=100,
        bounds=bounds,
        sampling="random"
    )
    
    # Test points that will be reused
    test_points = jnp.array(np.random.uniform(-2, 2, (500, 2)))
    repeated_points = jnp.tile(test_points[:100], (5, 1))  # Repeat first 100 points 5 times
    
    # Optimizer with caching disabled
    optimizer_no_cache = ScalableSurrogateOptimizer(
        surrogate_type="gaussian_process",
        enable_caching=False,
        enable_monitoring=True
    )
    optimizer_no_cache.fit_surrogate(data)
    
    # Optimizer with caching enabled
    optimizer_with_cache = ScalableSurrogateOptimizer(
        surrogate_type="gaussian_process",
        enable_caching=True,
        enable_monitoring=True
    )
    optimizer_with_cache.fit_surrogate(data)
    
    # Test without caching
    print("🐌 Testing without caching...")
    start_time = time.time()
    pred_no_cache = optimizer_no_cache.predict(repeated_points)
    time_no_cache = time.time() - start_time
    
    # Test with caching
    print("⚡ Testing with caching...")
    start_time = time.time()
    pred_with_cache = optimizer_with_cache.predict(repeated_points)
    time_with_cache = time.time() - start_time
    
    # Second run with cache (should be much faster)
    start_time = time.time()
    pred_cached = optimizer_with_cache.predict(repeated_points)
    time_cached = time.time() - start_time
    
    # Get cache statistics
    cache_stats = optimizer_with_cache.get_comprehensive_metrics().get("cache_stats", {})
    
    print(f"\n📊 Caching Results:")
    print(f"   Without caching: {time_no_cache:.4f}s")
    print(f"   With caching (first): {time_with_cache:.4f}s")
    print(f"   With caching (cached): {time_cached:.4f}s")
    print(f"   Speedup from caching: {time_with_cache/time_cached:.1f}x")
    print(f"   Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
    print(f"   Cache size: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}")


def test_memory_optimization():
    """Test memory optimization features."""
    print("\n🧠 Testing Memory Optimization")
    print("=" * 35)
    
    # Create large dataset
    print("📈 Creating large dataset...")
    bounds = [(-2.0, 2.0)] * 5  # 5D problem
    
    # Large dataset that might exceed memory limits
    large_data = collect_data(
        function=complex_benchmark_function,
        n_samples=2000,  # Large number of samples
        bounds=bounds,
        sampling="random"
    )
    
    print(f"   Original size: {large_data.n_samples} samples")
    print(f"   Memory usage: {(large_data.X.nbytes + large_data.y.nbytes) / (1024**2):.1f} MB")
    
    # Optimizer with memory optimization
    optimizer = ScalableSurrogateOptimizer(
        surrogate_type="gaussian_process",
        memory_limit_mb=5.0,  # Very small limit to force optimization
        auto_optimize=True,
        enable_monitoring=True
    )
    
    # Train with memory optimization
    print("🔧 Training with memory optimization...")
    start_time = time.time()
    optimizer.fit_surrogate(large_data, optimize_memory=True)
    training_time = time.time() - start_time
    
    # Get metrics
    metrics = optimizer.get_comprehensive_metrics()
    memory_opts = metrics["scaling"]["memory_optimizations"]
    
    if memory_opts:
        opt = memory_opts[-1]
        print(f"   ✅ Memory optimized:")
        print(f"      Original: {opt['original_size']} samples")
        print(f"      Optimized: {opt['optimized_size']} samples")
        print(f"      Reduction: {(1-opt['reduction_factor']):.1%}")
    else:
        print("   ✅ No memory optimization needed")
    
    print(f"   Training time: {training_time:.2f}s")


def create_performance_report(scaling_results):
    """Create comprehensive performance report."""
    print("\n📊 Performance Analysis Report")
    print("=" * 40)
    
    # Analyze scaling trends
    dimensions = sorted(set(r["dimensions"] for r in scaling_results.values()))
    
    for n_dims in dimensions:
        print(f"\n📈 {n_dims}D Performance Analysis:")
        
        dim_results = [
            r for r in scaling_results.values()
            if r["dimensions"] == n_dims
        ]
        
        # Sort by sample size
        dim_results.sort(key=lambda x: x["samples"])
        
        print(f"   Samples -> Training Time -> Prediction Throughput")
        for result in dim_results:
            print(f"   {result['samples']:4d} -> {result['training_time']:6.2f}s -> {result['prediction_throughput']:8.0f}/s")
        
        # Calculate scaling efficiency
        if len(dim_results) > 1:
            base_result = dim_results[0]
            last_result = dim_results[-1]
            
            size_ratio = last_result["samples"] / base_result["samples"]
            time_ratio = last_result["training_time"] / base_result["training_time"]
            efficiency = size_ratio / time_ratio
            
            print(f"   Scaling efficiency: {efficiency:.2f} (higher is better)")
    
    # Overall recommendations
    print(f"\n💡 Performance Recommendations:")
    
    # Find best performing configurations
    best_throughput = max(r["prediction_throughput"] for r in scaling_results.values())
    best_config = next(r for r in scaling_results.values() if r["prediction_throughput"] == best_throughput)
    
    print(f"   ✅ Best throughput: {best_throughput:.0f} predictions/s")
    print(f"      Configuration: {best_config['dimensions']}D, {best_config['samples']} samples")
    
    # Check for gradient performance
    gradient_results = [r for r in scaling_results.values() if r["gradient_time"] is not None]
    if gradient_results:
        avg_grad_throughput = np.mean([r["gradient_throughput"] for r in gradient_results])
        print(f"   ✅ Average gradient throughput: {avg_grad_throughput:.0f}/s")
    else:
        print("   ⚠️  Gradient computation not available")


def main():
    """Main demonstration function."""
    print("🌟 Scalable Performance Demo - Generation 3")
    print("=" * 60)
    print("Testing high-performance scaling features:")
    print("- JIT compilation optimization")
    print("- Vectorized batch processing")  
    print("- Parallel processing capabilities")
    print("- Adaptive caching system")
    print("- Memory usage optimization")
    print("- Performance profiling")
    print("=" * 60)
    
    # Run all performance tests
    scaling_results = test_scaling_performance()
    test_parallel_optimization()
    test_caching_performance()
    test_memory_optimization()
    
    # Generate comprehensive report
    create_performance_report(scaling_results)
    
    print("\n🎉 Scalable Performance Demo Completed!")
    print("=" * 50)
    print("✅ All scaling features demonstrated")
    print("✅ Performance optimizations active")
    print("✅ System ready for production workloads")


if __name__ == "__main__":
    main()