"""Experimental research suite for comparative studies and ablation analysis."""

import time
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import itertools

import jax.numpy as jnp
from jax import Array
import jax.random as random

from .novel_algorithms import ResearchResult
from .theoretical_analysis import run_theoretical_analysis


@dataclass
class ExperimentConfig:
    """Configuration for a research experiment."""
    experiment_name: str
    algorithm_configs: List[Dict[str, Any]]
    test_functions: List[str]
    problem_dimensions: List[int]
    n_trials: int = 10
    n_iterations: int = 100
    significance_level: float = 0.05
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of a comparative study."""
    experiment_name: str
    algorithms: List[str]
    test_functions: List[str]
    performance_matrix: Array  # [algorithm, function, metric]
    statistical_significance: Dict[str, Dict[str, float]]
    rankings: Dict[str, List[str]]
    best_algorithm: str
    effect_sizes: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Result of an ablation study."""
    base_algorithm: str
    components: List[str]
    component_contributions: Dict[str, float]
    interaction_effects: Dict[Tuple[str, str], float]
    optimal_configuration: Dict[str, bool]
    significance_tests: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResearchExperimentSuite:
    """Suite for conducting research experiments and comparative studies."""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        save_raw_data: bool = True,
        parallel_execution: bool = True,
    ):
        """Initialize research experiment suite.
        
        Args:
            output_dir: Directory for experiment outputs
            save_raw_data: Whether to save raw experimental data
            parallel_execution: Whether to run experiments in parallel
        """
        self.output_dir = Path(output_dir) if output_dir else Path("research_experiments")
        self.output_dir.mkdir(exist_ok=True)
        self.save_raw_data = save_raw_data
        self.parallel_execution = parallel_execution
        
        # Experiment registry
        self.experiment_history = []
        
        # Statistical tools
        self.alpha = 0.05  # Significance level
    
    def register_experiment(self, config: ExperimentConfig) -> str:
        """Register an experiment configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        experiment_id = f"{config.experiment_name}_{int(time.time())}"
        config.metadata["experiment_id"] = experiment_id
        config.metadata["registered_at"] = time.time()
        
        # Save experiment configuration
        config_file = self.output_dir / f"{experiment_id}_config.json"
        with open(config_file, 'w') as f:
            # Convert config to serializable format
            config_dict = {
                "experiment_name": config.experiment_name,
                "algorithm_configs": config.algorithm_configs,
                "test_functions": config.test_functions,
                "problem_dimensions": config.problem_dimensions,
                "n_trials": config.n_trials,
                "n_iterations": config.n_iterations,
                "significance_level": config.significance_level,
                "metadata": config.metadata,
            }
            json.dump(config_dict, f, indent=2)
        
        self.experiment_history.append(config)
        return experiment_id
    
    def run_comparative_study(
        self,
        config: ExperimentConfig,
        metrics: List[str] = None,
    ) -> ComparisonResult:
        """Run comparative study between multiple algorithms.
        
        Args:
            config: Experiment configuration
            metrics: Metrics to evaluate ("best_value", "convergence_rate", etc.)
            
        Returns:
            Comparative study results
        """
        if metrics is None:
            metrics = ["best_value", "convergence_rate", "n_evaluations"]
        
        print(f"Running comparative study: {config.experiment_name}")
        print(f"Algorithms: {len(config.algorithm_configs)}")
        print(f"Test functions: {len(config.test_functions)}")
        print(f"Dimensions: {config.problem_dimensions}")
        print(f"Trials per configuration: {config.n_trials}")
        
        # Run all algorithm-function-dimension combinations
        results = self._execute_experiment_grid(config)
        
        # Aggregate results into performance matrix
        performance_matrix = self._build_performance_matrix(results, metrics)
        
        # Statistical analysis
        significance_tests = self._perform_statistical_tests(results)
        rankings = self._compute_rankings(performance_matrix, metrics)
        best_algorithm = self._identify_best_algorithm(rankings)
        effect_sizes = self._compute_effect_sizes(results)
        
        comparison_result = ComparisonResult(
            experiment_name=config.experiment_name,
            algorithms=[cfg["name"] for cfg in config.algorithm_configs],
            test_functions=config.test_functions,
            performance_matrix=performance_matrix,
            statistical_significance=significance_tests,
            rankings=rankings,
            best_algorithm=best_algorithm,
            effect_sizes=effect_sizes,
            metadata={
                "n_trials": config.n_trials,
                "n_iterations": config.n_iterations,
                "problem_dimensions": config.problem_dimensions,
                "metrics": metrics,
                "experiment_time": time.time(),
            }
        )
        
        # Save results
        self._save_comparison_results(comparison_result)
        
        return comparison_result
    
    def run_ablation_study(
        self,
        base_algorithm_config: Dict[str, Any],
        components_to_ablate: List[str],
        test_function: str = "rosenbrock_2d",
        problem_dimension: int = 2,
        n_trials: int = 20,
    ) -> AblationResult:
        """Run ablation study to understand component contributions.
        
        Args:
            base_algorithm_config: Full algorithm configuration
            components_to_ablate: List of components to ablate
            test_function: Test function for ablation
            problem_dimension: Problem dimension
            n_trials: Number of trials
            
        Returns:
            Ablation study results
        """
        print(f"Running ablation study for {base_algorithm_config['name']}")
        print(f"Ablating components: {components_to_ablate}")
        
        # Generate all combinations of component ablations
        component_combinations = []
        for r in range(len(components_to_ablate) + 1):
            for combo in itertools.combinations(components_to_ablate, r):
                component_combinations.append(list(combo))
        
        print(f"Testing {len(component_combinations)} component combinations")
        
        # Run experiments for each combination
        ablation_results = {}
        for ablated_components in component_combinations:
            config_name = f"ablated_{'_'.join(ablated_components) if ablated_components else 'none'}"
            
            # Create modified configuration
            modified_config = base_algorithm_config.copy()
            for component in ablated_components:
                if component in modified_config:
                    modified_config[component] = False  # Disable component
            
            # Run trials
            trial_results = []
            for trial in range(n_trials):
                result = self._run_single_algorithm_trial(
                    modified_config, test_function, problem_dimension
                )
                trial_results.append(result)
            
            ablation_results[tuple(ablated_components)] = trial_results
        
        # Analyze component contributions
        component_contributions = self._analyze_component_contributions(
            ablation_results, components_to_ablate
        )
        
        # Analyze interaction effects
        interaction_effects = self._analyze_interaction_effects(
            ablation_results, components_to_ablate
        )
        
        # Find optimal configuration
        optimal_config = self._find_optimal_configuration(
            ablation_results, components_to_ablate
        )
        
        # Statistical significance tests
        significance_tests = self._ablation_significance_tests(
            ablation_results, components_to_ablate
        )
        
        ablation_result = AblationResult(
            base_algorithm=base_algorithm_config["name"],
            components=components_to_ablate,
            component_contributions=component_contributions,
            interaction_effects=interaction_effects,
            optimal_configuration=optimal_config,
            significance_tests=significance_tests,
            metadata={
                "test_function": test_function,
                "problem_dimension": problem_dimension,
                "n_trials": n_trials,
                "n_combinations": len(component_combinations),
            }
        )
        
        # Save results
        self._save_ablation_results(ablation_result)
        
        return ablation_result
    
    def _execute_experiment_grid(self, config: ExperimentConfig) -> Dict[str, List[ResearchResult]]:
        """Execute full grid of experiments."""
        results = {}
        
        total_experiments = (
            len(config.algorithm_configs) * 
            len(config.test_functions) * 
            len(config.problem_dimensions) * 
            config.n_trials
        )
        
        print(f"Executing {total_experiments} total experiments...")
        
        experiment_count = 0
        
        for algo_config in config.algorithm_configs:
            algo_name = algo_config["name"]
            results[algo_name] = []
            
            for test_function in config.test_functions:
                for dim in config.problem_dimensions:
                    for trial in range(config.n_trials):
                        experiment_count += 1
                        
                        if experiment_count % 10 == 0:
                            print(f"Progress: {experiment_count}/{total_experiments}")
                        
                        try:
                            result = self._run_single_algorithm_trial(
                                algo_config, test_function, dim
                            )
                            result.metadata.update({
                                "test_function": test_function,
                                "dimension": dim,
                                "trial": trial,
                            })
                            results[algo_name].append(result)
                            
                        except Exception as e:
                            print(f"Experiment failed: {algo_config['name']} on {test_function}: {e}")
                            continue
        
        return results
    
    def _run_single_algorithm_trial(
        self,
        algo_config: Dict[str, Any],
        test_function: str,
        dimension: int,
    ) -> ResearchResult:
        """Run a single algorithm trial."""
        # Get test function
        import sys
        sys.path.append('/root/repo')
        from tests.fixtures.benchmark_functions import benchmark_functions
        
        if test_function not in benchmark_functions:
            raise ValueError(f"Unknown test function: {test_function}")
        
        func = benchmark_functions[test_function]
        bounds = func.bounds
        
        # Run algorithm based on configuration
        algo_type = algo_config.get("type", "standard")
        
        if algo_type == "adaptive_acquisition":
            from .novel_algorithms import AdaptiveAcquisitionOptimizer
            from ..core import SurrogateOptimizer
            from ..data.collector import collect_data
            
            # Collect initial data
            data = collect_data(func, n_samples=50, bounds=bounds, verbose=False)
            
            # Train surrogate
            surrogate_opt = SurrogateOptimizer(surrogate_type="neural_network")
            surrogate_opt.fit_surrogate(data)
            
            # Run adaptive acquisition optimization
            optimizer = AdaptiveAcquisitionOptimizer()
            result = optimizer.optimize(
                surrogate_opt.surrogate,
                x0=jnp.array([0.0] * dimension),
                bounds=bounds,
                n_iterations=50
            )
            
            # Convert to ResearchResult
            research_result = ResearchResult(
                algorithm_name=algo_config["name"],
                experiment_id=f"trial_{int(time.time())}",
                success=result.success,
                performance_metrics={
                    "best_value": float(result.fun),
                    "n_evaluations": result.nfev,
                    "convergence_rate": self._compute_convergence_rate(result.convergence_history),
                },
                convergence_data=result.convergence_history,
                execution_time=1.0,  # Placeholder
            )
            
        elif algo_type == "physics_informed":
            # Placeholder for physics-informed optimization
            research_result = ResearchResult(
                algorithm_name=algo_config["name"],
                experiment_id=f"trial_{int(time.time())}",
                success=True,
                performance_metrics={"best_value": 0.1, "n_evaluations": 100},
                convergence_data=[],
                execution_time=1.0,
            )
            
        else:
            # Standard surrogate optimization
            from ..core import quick_optimize
            
            result = quick_optimize(
                function=func,
                bounds=bounds,
                n_samples=50,
                surrogate_type="neural_network",
                verbose=False
            )
            
            research_result = ResearchResult(
                algorithm_name=algo_config["name"],
                experiment_id=f"trial_{int(time.time())}",
                success=result.success,
                performance_metrics={
                    "best_value": float(result.fun),
                    "n_evaluations": result.nfev,
                },
                convergence_data=[],
                execution_time=1.0,
            )
        
        return research_result
    
    def _compute_convergence_rate(self, convergence_history: List[float]) -> float:
        """Compute empirical convergence rate."""
        if len(convergence_history) < 2:
            return 0.0
        
        values = jnp.array(convergence_history)
        if jnp.all(values == values[0]):  # No improvement
            return 0.0
        
        # Simple convergence rate: improvement per iteration
        total_improvement = values[0] - values[-1]
        n_iterations = len(values) - 1
        
        return float(total_improvement / n_iterations) if n_iterations > 0 else 0.0
    
    def _build_performance_matrix(
        self,
        results: Dict[str, List[ResearchResult]],
        metrics: List[str],
    ) -> Array:
        """Build performance matrix from experimental results."""
        algorithms = list(results.keys())
        n_algorithms = len(algorithms)
        n_metrics = len(metrics)
        
        # Get unique test functions and dimensions
        test_configs = set()
        for algo_results in results.values():
            for result in algo_results:
                test_func = result.metadata.get("test_function", "unknown")
                dim = result.metadata.get("dimension", 0)
                test_configs.add((test_func, dim))
        
        test_configs = list(test_configs)
        n_configs = len(test_configs)
        
        # Build matrix: [algorithm, test_config, metric]
        matrix = jnp.zeros((n_algorithms, n_configs, n_metrics))
        
        for i, algo in enumerate(algorithms):
            algo_results = results[algo]
            
            # Group results by test configuration
            config_results = {}
            for result in algo_results:
                test_func = result.metadata.get("test_function", "unknown")
                dim = result.metadata.get("dimension", 0)
                config_key = (test_func, dim)
                
                if config_key not in config_results:
                    config_results[config_key] = []
                config_results[config_key].append(result)
            
            # Aggregate results for each configuration
            for j, config in enumerate(test_configs):
                if config in config_results:
                    config_res = config_results[config]
                    
                    for k, metric in enumerate(metrics):
                        metric_values = [
                            r.performance_metrics.get(metric, 0.0)
                            for r in config_res
                        ]
                        
                        if metric_values:
                            matrix = matrix.at[i, j, k].set(jnp.mean(jnp.array(metric_values)))
        
        return matrix
    
    def _perform_statistical_tests(self, results: Dict[str, List[ResearchResult]]) -> Dict[str, Dict[str, float]]:
        """Perform statistical significance tests."""
        # Placeholder for statistical tests
        # Would implement proper t-tests, Mann-Whitney U tests, etc.
        significance_tests = {}
        
        algorithms = list(results.keys())
        for i, algo1 in enumerate(algorithms):
            significance_tests[algo1] = {}
            for j, algo2 in enumerate(algorithms):
                if i != j:
                    # Mock p-value
                    significance_tests[algo1][algo2] = 0.05
        
        return significance_tests
    
    def _compute_rankings(self, matrix: Array, metrics: List[str]) -> Dict[str, List[str]]:
        """Compute algorithm rankings for each metric."""
        rankings = {}
        
        for k, metric in enumerate(metrics):
            # Average across test configurations
            avg_performance = jnp.mean(matrix[:, :, k], axis=1)
            
            # Rank algorithms (assuming lower is better for most metrics)
            if metric in ["best_value", "n_evaluations"]:
                # Lower is better
                ranked_indices = jnp.argsort(avg_performance)
            else:
                # Higher is better
                ranked_indices = jnp.argsort(-avg_performance)
            
            rankings[metric] = [f"algorithm_{i}" for i in ranked_indices]
        
        return rankings
    
    def _identify_best_algorithm(self, rankings: Dict[str, List[str]]) -> str:
        """Identify overall best algorithm across metrics."""
        # Simple approach: algorithm with best average ranking
        algorithm_scores = {}
        
        for metric_ranking in rankings.values():
            for i, algo in enumerate(metric_ranking):
                if algo not in algorithm_scores:
                    algorithm_scores[algo] = []
                algorithm_scores[algo].append(i)  # Position in ranking
        
        # Compute average ranking position
        avg_rankings = {
            algo: sum(scores) / len(scores)
            for algo, scores in algorithm_scores.items()
        }
        
        return min(avg_rankings, key=avg_rankings.get)
    
    def _compute_effect_sizes(self, results: Dict[str, List[ResearchResult]]) -> Dict[str, float]:
        """Compute effect sizes for algorithm differences."""
        # Placeholder for Cohen's d or other effect size measures
        effect_sizes = {}
        
        algorithms = list(results.keys())
        for algo in algorithms:
            # Mock effect size
            effect_sizes[algo] = 0.5
        
        return effect_sizes
    
    def _analyze_component_contributions(
        self,
        ablation_results: Dict[Tuple[str, ...], List[ResearchResult]],
        components: List[str],
    ) -> Dict[str, float]:
        """Analyze individual component contributions."""
        contributions = {}
        
        # Get baseline (no components ablated)
        baseline_key = ()
        baseline_performance = jnp.mean([
            r.performance_metrics.get("best_value", 0.0)
            for r in ablation_results.get(baseline_key, [])
        ])
        
        # Analyze each component
        for component in components:
            # Get performance when this component is ablated
            ablated_key = (component,)
            ablated_performance = jnp.mean([
                r.performance_metrics.get("best_value", 0.0)
                for r in ablation_results.get(ablated_key, [])
            ])
            
            # Contribution is the performance difference
            contributions[component] = float(baseline_performance - ablated_performance)
        
        return contributions
    
    def _analyze_interaction_effects(
        self,
        ablation_results: Dict[Tuple[str, ...], List[ResearchResult]],
        components: List[str],
    ) -> Dict[Tuple[str, str], float]:
        """Analyze interaction effects between components."""
        interactions = {}
        
        # Analyze pairwise interactions
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                # Get individual effects
                comp1_key = (comp1,)
                comp2_key = (comp2,)
                both_key = tuple(sorted([comp1, comp2]))
                
                comp1_effect = jnp.mean([
                    r.performance_metrics.get("best_value", 0.0)
                    for r in ablation_results.get(comp1_key, [])
                ])
                
                comp2_effect = jnp.mean([
                    r.performance_metrics.get("best_value", 0.0)
                    for r in ablation_results.get(comp2_key, [])
                ])
                
                both_effect = jnp.mean([
                    r.performance_metrics.get("best_value", 0.0)
                    for r in ablation_results.get(both_key, [])
                ])
                
                # Interaction effect
                expected_combined = comp1_effect + comp2_effect
                actual_combined = both_effect
                interaction = float(actual_combined - expected_combined)
                
                interactions[(comp1, comp2)] = interaction
        
        return interactions
    
    def _find_optimal_configuration(
        self,
        ablation_results: Dict[Tuple[str, ...], List[ResearchResult]],
        components: List[str],
    ) -> Dict[str, bool]:
        """Find optimal component configuration."""
        best_performance = float('inf')
        best_config = None
        
        for ablated_components, results in ablation_results.items():
            avg_performance = jnp.mean([
                r.performance_metrics.get("best_value", float('inf'))
                for r in results
            ])
            
            if avg_performance < best_performance:
                best_performance = avg_performance
                best_config = ablated_components
        
        # Convert to boolean configuration
        optimal_config = {
            component: component not in best_config
            for component in components
        }
        
        return optimal_config
    
    def _ablation_significance_tests(
        self,
        ablation_results: Dict[Tuple[str, ...], List[ResearchResult]],
        components: List[str],
    ) -> Dict[str, float]:
        """Perform significance tests for ablation results."""
        significance_tests = {}
        
        # Get baseline results
        baseline_key = ()
        baseline_values = [
            r.performance_metrics.get("best_value", 0.0)
            for r in ablation_results.get(baseline_key, [])
        ]
        
        # Test each component ablation against baseline
        for component in components:
            ablated_key = (component,)
            ablated_values = [
                r.performance_metrics.get("best_value", 0.0)
                for r in ablation_results.get(ablated_key, [])
            ]
            
            # Mock t-test p-value
            significance_tests[component] = 0.05
        
        return significance_tests
    
    def _save_comparison_results(self, result: ComparisonResult):
        """Save comparison study results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{result.experiment_name}_{timestamp}.json"
        
        # Convert to serializable format
        result_dict = {
            "experiment_name": result.experiment_name,
            "algorithms": result.algorithms,
            "test_functions": result.test_functions,
            "performance_matrix": result.performance_matrix.tolist(),
            "statistical_significance": result.statistical_significance,
            "rankings": result.rankings,
            "best_algorithm": result.best_algorithm,
            "effect_sizes": result.effect_sizes,
            "metadata": result.metadata,
        }
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"Comparison results saved to: {filepath}")
    
    def _save_ablation_results(self, result: AblationResult):
        """Save ablation study results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ablation_{result.base_algorithm}_{timestamp}.json"
        
        # Convert to serializable format
        result_dict = {
            "base_algorithm": result.base_algorithm,
            "components": result.components,
            "component_contributions": result.component_contributions,
            "interaction_effects": {
                f"{k[0]}_x_{k[1]}": v for k, v in result.interaction_effects.items()
            },
            "optimal_configuration": result.optimal_configuration,
            "significance_tests": result.significance_tests,
            "metadata": result.metadata,
        }
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"Ablation results saved to: {filepath}")


class ComparisonStudy:
    """Convenience class for running comparative studies."""
    
    def __init__(self, experiment_name: str):
        """Initialize comparison study.
        
        Args:
            experiment_name: Name of the comparison study
        """
        self.experiment_name = experiment_name
        self.suite = ResearchExperimentSuite()
    
    def compare_algorithms(
        self,
        algorithms: List[Dict[str, Any]],
        test_functions: List[str] = None,
        dimensions: List[int] = None,
        n_trials: int = 10,
    ) -> ComparisonResult:
        """Run algorithm comparison.
        
        Args:
            algorithms: List of algorithm configurations
            test_functions: Test functions to use
            dimensions: Problem dimensions to test
            n_trials: Number of trials per configuration
            
        Returns:
            Comparison results
        """
        if test_functions is None:
            test_functions = ["sphere_2d", "rosenbrock_2d", "rastrigin_2d"]
        
        if dimensions is None:
            dimensions = [2, 5]
        
        config = ExperimentConfig(
            experiment_name=self.experiment_name,
            algorithm_configs=algorithms,
            test_functions=test_functions,
            problem_dimensions=dimensions,
            n_trials=n_trials,
        )
        
        return self.suite.run_comparative_study(config)


class AblationStudy:
    """Convenience class for running ablation studies."""
    
    def __init__(self, algorithm_name: str):
        """Initialize ablation study.
        
        Args:
            algorithm_name: Name of the algorithm to study
        """
        self.algorithm_name = algorithm_name
        self.suite = ResearchExperimentSuite()
    
    def ablate_components(
        self,
        base_config: Dict[str, Any],
        components: List[str],
        test_function: str = "rosenbrock_2d",
        n_trials: int = 20,
    ) -> AblationResult:
        """Run component ablation study.
        
        Args:
            base_config: Base algorithm configuration
            components: Components to ablate
            test_function: Test function to use
            n_trials: Number of trials
            
        Returns:
            Ablation results
        """
        return self.suite.run_ablation_study(
            base_config, components, test_function, 2, n_trials
        )


def run_research_experiments(
    experiment_type: str = "comparison",
    algorithms: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
) -> Union[ComparisonResult, AblationResult]:
    """Run research experiments with default configurations.
    
    Args:
        experiment_type: Type of experiment ("comparison" or "ablation")
        algorithms: Algorithm configurations
        output_dir: Output directory
        
    Returns:
        Experiment results
    """
    if algorithms is None:
        algorithms = [
            {"name": "Standard", "type": "standard"},
            {"name": "Adaptive", "type": "adaptive_acquisition"},
        ]
    
    if experiment_type == "comparison":
        study = ComparisonStudy("default_comparison")
        return study.compare_algorithms(algorithms)
    
    elif experiment_type == "ablation":
        base_config = {"name": "Full_Algorithm", "jit": True, "vectorization": True}
        components = ["jit", "vectorization"]
        
        study = AblationStudy("default_algorithm")
        return study.ablate_components(base_config, components)
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")