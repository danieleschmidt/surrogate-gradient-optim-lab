#!/usr/bin/env python3
"""
Enhanced Integration Example for Surrogate Gradient Optimization Lab
Demonstrates advanced features and autonomous SDLC integration
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path

# Core imports
from surrogate_optim import SurrogateOptimizer, collect_data
from surrogate_optim.models import NeuralSurrogate, GPSurrogate, HybridSurrogate
from surrogate_optim.visualization import basic_plots


class EnhancedSurrogateDemo:
    """Enhanced demonstration of surrogate optimization capabilities."""
    
    def __init__(self, save_plots: bool = True):
        self.save_plots = save_plots
        self.results = {}
        self.plots_dir = Path("plots")
        if save_plots:
            self.plots_dir.mkdir(exist_ok=True)
    
    def complex_test_function(self, x: jnp.ndarray) -> float:
        """Complex multi-modal test function with local optima."""
        # Combination of Rosenbrock and Rastrigin characteristics
        x = jnp.atleast_1d(x)
        
        # Rosenbrock-like component
        rosenbrock = jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        # Rastrigin-like component for multi-modality
        rastrigin = 10 * len(x) + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))
        
        # Add some noise and nonlinearity
        noise_term = 0.1 * jnp.sum(jnp.sin(5 * x) * jnp.exp(-0.1 * x**2))
        
        return float(0.3 * rosenbrock + 0.2 * rastrigin + noise_term)
    
    def engineering_design_function(self, x: jnp.ndarray) -> float:
        """Engineering design optimization problem (pressure vessel)."""
        x = jnp.atleast_1d(x)
        if len(x) != 4:
            raise ValueError("Pressure vessel problem requires 4 variables")
        
        # Variables: thickness of shell, thickness of head, inner radius, length
        ts, th, R, L = x[0], x[1], x[2], x[3]
        
        # Objective: minimize cost
        cost = (0.6224 * ts * R * L + 
                1.7781 * th * R**2 + 
                3.1661 * ts**2 * L + 
                19.84 * ts**2 * R)
        
        # Add penalty for constraint violations
        penalty = 0
        if -ts + 0.0193 * R > 0:
            penalty += 1000 * (-ts + 0.0193 * R)**2
        if -th + 0.00954 * R > 0:
            penalty += 1000 * (-th + 0.00954 * R)**2
        if -jnp.pi * R**2 * L - (4/3) * jnp.pi * R**3 + 1296000 > 0:
            penalty += 1000 * (-jnp.pi * R**2 * L - (4/3) * jnp.pi * R**3 + 1296000)**2
        if L - 240 > 0:
            penalty += 1000 * (L - 240)**2
        
        return float(cost + penalty)
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark comparing different surrogate types."""
        print("🚀 Starting Enhanced Surrogate Optimization Benchmark")
        print("=" * 60)
        
        # Test functions and their properties
        test_cases = [
            {
                "name": "Complex Multi-Modal 2D",
                "function": self.complex_test_function,
                "bounds": [(-5.0, 5.0)] * 2,
                "optimal_point": jnp.array([1.0, 1.0]),
                "n_samples": 200,
                "n_dims": 2
            },
            {
                "name": "Engineering Design 4D",
                "function": self.engineering_design_function,
                "bounds": [(0.1, 2.0), (0.1, 2.0), (10.0, 200.0), (10.0, 200.0)],
                "optimal_point": None,  # Unknown optimal
                "n_samples": 400,
                "n_dims": 4
            }
        ]
        
        # Surrogate configurations
        surrogate_configs = [
            {
                "name": "Neural Network",
                "type": "neural_network",
                "params": {"hidden_dims": [128, 64, 32], "activation": "gelu", "learning_rate": 0.001}
            },
            {
                "name": "Gaussian Process",
                "type": "gaussian_process", 
                "params": {"kernel": "rbf", "length_scale": 1.0, "noise_level": 0.01}
            },
            {
                "name": "Hybrid Model",
                "type": "hybrid",
                "params": {}  # Use defaults
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            print(f"\n📊 Testing: {test_case['name']}")
            print("-" * 40)
            
            case_results = {}
            
            # Collect training data once
            print(f"📈 Collecting {test_case['n_samples']} training samples...")
            start_time = time.time()
            
            data = collect_data(
                function=test_case['function'],
                n_samples=test_case['n_samples'],
                bounds=test_case['bounds'],
                sampling="sobol",
                verbose=False
            )
            
            data_collection_time = time.time() - start_time
            print(f"⏱️  Data collection time: {data_collection_time:.2f}s")
            
            # Test each surrogate type
            for config in surrogate_configs:
                print(f"\n🧠 Training {config['name']} surrogate...")
                
                # Create optimizer
                optimizer = SurrogateOptimizer(
                    surrogate_type=config['type'],
                    surrogate_params=config['params'],
                    optimizer_type="multi_start",
                    optimizer_params={"n_starts": 5}
                )
                
                # Train surrogate
                start_time = time.time()
                optimizer.fit_surrogate(data)
                training_time = time.time() - start_time
                
                # Run optimization
                initial_point = jnp.array([
                    (bounds[0] + bounds[1]) / 2 for bounds in test_case['bounds']
                ])
                
                start_time = time.time()
                result = optimizer.optimize(
                    initial_point=initial_point,
                    bounds=test_case['bounds'],
                    num_steps=100
                )
                optimization_time = time.time() - start_time
                
                # Validate results
                validation_metrics = optimizer.validate(
                    test_function=test_case['function'],
                    n_test_points=100,
                    metrics=["mse", "mae", "r2", "gradient_error"]
                )
                
                # Store results
                case_results[config['name']] = {
                    "training_time": training_time,
                    "optimization_time": optimization_time,
                    "final_value": float(result.fun),
                    "optimal_point": result.x.tolist(),
                    "n_iterations": result.nit if hasattr(result, 'nit') else None,
                    "validation_metrics": validation_metrics,
                    "success": result.success if hasattr(result, 'success') else True
                }
                
                print(f"   ✅ Training time: {training_time:.2f}s")
                print(f"   🎯 Optimization time: {optimization_time:.2f}s") 
                print(f"   📊 Final value: {result.fun:.6f}")
                print(f"   📍 Optimal point: {result.x}")
                print(f"   📈 Validation R²: {validation_metrics.get('r2', 'N/A'):.4f}")
            
            case_results["data_collection_time"] = data_collection_time
            case_results["n_training_samples"] = test_case['n_samples']
            results[test_case['name']] = case_results
        
        self.results = results
        return results
    
    def create_performance_visualizations(self):
        """Create comprehensive performance visualizations."""
        if not self.results:
            print("❌ No results available. Run benchmark first.")
            return
        
        print("\n📊 Creating Performance Visualizations...")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Surrogate Optimization Performance Comparison", fontsize=16)
        
        # Extract data for plotting
        test_cases = list(self.results.keys())
        surrogate_types = ["Neural Network", "Gaussian Process", "Hybrid Model"]
        
        # Training time comparison
        ax = axes[0, 0]
        training_times = {}
        for test_case in test_cases:
            times = [self.results[test_case][stype]["training_time"] 
                    for stype in surrogate_types]
            training_times[test_case] = times
        
        x = np.arange(len(test_cases))
        width = 0.25
        for i, stype in enumerate(surrogate_types):
            times = [training_times[tc][i] for tc in test_cases]
            ax.bar(x + i*width, times, width, label=stype)
        
        ax.set_xlabel("Test Cases")
        ax.set_ylabel("Training Time (seconds)")
        ax.set_title("Training Time Comparison")
        ax.set_xticks(x + width)
        ax.set_xticklabels([tc[:15] + "..." if len(tc) > 15 else tc for tc in test_cases])
        ax.legend()
        ax.set_yscale('log')
        
        # Final objective values
        ax = axes[0, 1]
        final_values = {}
        for test_case in test_cases:
            values = [self.results[test_case][stype]["final_value"] 
                     for stype in surrogate_types]
            final_values[test_case] = values
        
        for i, stype in enumerate(surrogate_types):
            values = [final_values[tc][i] for tc in test_cases]
            ax.bar(x + i*width, values, width, label=stype)
        
        ax.set_xlabel("Test Cases")
        ax.set_ylabel("Final Objective Value")
        ax.set_title("Final Optimization Results")
        ax.set_xticks(x + width)
        ax.set_xticklabels([tc[:15] + "..." if len(tc) > 15 else tc for tc in test_cases])
        ax.legend()
        ax.set_yscale('log')
        
        # Validation R² scores
        ax = axes[1, 0]
        r2_scores = {}
        for test_case in test_cases:
            scores = [self.results[test_case][stype]["validation_metrics"].get("r2", 0) 
                     for stype in surrogate_types]
            r2_scores[test_case] = scores
        
        for i, stype in enumerate(surrogate_types):
            scores = [r2_scores[tc][i] for tc in test_cases]
            ax.bar(x + i*width, scores, width, label=stype)
        
        ax.set_xlabel("Test Cases")
        ax.set_ylabel("R² Score")
        ax.set_title("Surrogate Model Accuracy")
        ax.set_xticks(x + width)
        ax.set_xticklabels([tc[:15] + "..." if len(tc) > 15 else tc for tc in test_cases])
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Total time (training + optimization)
        ax = axes[1, 1]
        total_times = {}
        for test_case in test_cases:
            times = [self.results[test_case][stype]["training_time"] + 
                    self.results[test_case][stype]["optimization_time"]
                    for stype in surrogate_types]
            total_times[test_case] = times
        
        for i, stype in enumerate(surrogate_types):
            times = [total_times[tc][i] for tc in test_cases]
            ax.bar(x + i*width, times, width, label=stype)
        
        ax.set_xlabel("Test Cases")
        ax.set_ylabel("Total Time (seconds)")
        ax.set_title("Total Computation Time")
        ax.set_xticks(x + width)
        ax.set_xticklabels([tc[:15] + "..." if len(tc) > 15 else tc for tc in test_cases])
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plots_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
            print(f"📊 Performance plots saved to {self.plots_dir / 'performance_comparison.png'}")
        
        plt.show()
    
    def generate_performance_report(self) -> str:
        """Generate detailed performance report."""
        if not self.results:
            return "No results available. Run benchmark first."
        
        report = []
        report.append("# Surrogate Optimization Performance Report")
        report.append("=" * 50)
        report.append("")
        
        for test_case, case_results in self.results.items():
            report.append(f"## {test_case}")
            report.append("-" * len(test_case))
            report.append("")
            
            # Summary statistics
            report.append("### Summary Statistics")
            report.append(f"- Training samples: {case_results['n_training_samples']}")
            report.append(f"- Data collection time: {case_results['data_collection_time']:.2f}s")
            report.append("")
            
            # Surrogate comparison
            report.append("### Surrogate Model Comparison")
            report.append("")
            
            surrogate_types = [k for k in case_results.keys() if k not in ['data_collection_time', 'n_training_samples']]
            
            for stype in surrogate_types:
                results = case_results[stype]
                report.append(f"#### {stype}")
                report.append(f"- Training time: {results['training_time']:.3f}s")
                report.append(f"- Optimization time: {results['optimization_time']:.3f}s")
                report.append(f"- Final objective value: {results['final_value']:.6f}")
                report.append(f"- Validation R²: {results['validation_metrics'].get('r2', 'N/A'):.4f}")
                report.append(f"- Validation MSE: {results['validation_metrics'].get('mse', 'N/A'):.4f}")
                report.append(f"- Success: {results['success']}")
                report.append("")
            
            report.append("")
        
        report_text = "\n".join(report)
        
        if self.save_plots:
            with open(self.plots_dir / "performance_report.md", "w") as f:
                f.write(report_text)
            print(f"📄 Performance report saved to {self.plots_dir / 'performance_report.md'}")
        
        return report_text


def main():
    """Main demonstration function."""
    print("🌟 Enhanced Surrogate Gradient Optimization Lab Demo")
    print("=" * 60)
    
    # Create demo instance
    demo = EnhancedSurrogateDemo(save_plots=True)
    
    # Run comprehensive benchmark
    results = demo.run_comprehensive_benchmark()
    
    # Create visualizations
    demo.create_performance_visualizations()
    
    # Generate report
    report = demo.generate_performance_report()
    print("\n📄 Performance Report Summary:")
    print("-" * 30)
    print(report[:1000] + "..." if len(report) > 1000 else report)
    
    print("\n✅ Enhanced demonstration completed successfully!")
    print(f"📊 Results saved to: {demo.plots_dir}")


if __name__ == "__main__":
    main()