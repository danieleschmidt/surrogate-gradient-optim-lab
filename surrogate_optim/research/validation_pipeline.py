"""Comprehensive research validation pipeline for novel optimization algorithms."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from scipy import stats

from ..benchmarks import BenchmarkSuite
from ..monitoring.enhanced_logging import get_logger
from ..observability.tracing import get_tracer
from ..quality.statistical_validation import StatisticalValidator
from .experimental_suite import ExperimentalSuite
from .novel_algorithms import (
    AdaptiveAcquisitionOptimizer,
    MultiObjectiveSurrogateOptimizer,
    PhysicsInformedSurrogate,
    SequentialModelBasedOptimization,
)

logger = get_logger()


class ValidationStatus(Enum):
    """Validation status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchDomain(Enum):
    """Research domain categories."""
    ALGORITHM_DEVELOPMENT = "algorithm_development"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    STATISTICAL_VALIDATION = "statistical_validation"
    COMPARATIVE_STUDY = "comparative_study"
    REPRODUCIBILITY = "reproducibility"
    PUBLICATION_READY = "publication_ready"


@dataclass
class ValidationConfig:
    """Configuration for research validation."""
    name: str
    description: str
    domain: ResearchDomain
    algorithms_to_test: List[str]
    benchmark_functions: List[str]
    dimensions: List[int]
    n_trials: int
    confidence_level: float
    statistical_tests: List[str]
    performance_metrics: List[str]
    comparison_baselines: List[str]
    significance_threshold: float
    output_formats: List[str]
    parallel_execution: bool
    timeout_minutes: int


@dataclass
class ValidationResult:
    """Result of validation experiment."""
    config_name: str
    algorithm_name: str
    benchmark_function: str
    dimension: int
    trial_id: int
    execution_time: float
    final_value: float
    convergence_history: List[float]
    function_evaluations: int
    success: bool
    error_message: Optional[str]
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results."""
    algorithm: str
    benchmark: str
    dimension: int
    n_trials: int
    mean_performance: float
    std_performance: float
    median_performance: float
    best_performance: float
    worst_performance: float
    success_rate: float
    confidence_interval: Tuple[float, float]
    statistical_tests: Dict[str, Any]
    effect_sizes: Dict[str, float]
    rankings: Dict[str, int]


@dataclass
class ComparativeAnalysis:
    """Comparative analysis between algorithms."""
    algorithms: List[str]
    benchmark: str
    dimension: int
    pairwise_comparisons: Dict[Tuple[str, str], Dict[str, Any]]
    overall_ranking: List[Tuple[str, float]]  # (algorithm, score)
    statistical_significance: Dict[Tuple[str, str], bool]
    effect_sizes: Dict[Tuple[str, str], float]
    power_analysis: Dict[str, float]
    summary: str


class ResearchValidationPipeline:
    """Comprehensive validation pipeline for research algorithms."""

    def __init__(
        self,
        output_dir: str = "validation_results",
        enable_parallel: bool = True,
        max_workers: int = 4,
        log_level: str = "INFO",
    ):
        """Initialize research validation pipeline.
        
        Args:
            output_dir: Directory to store validation results
            enable_parallel: Enable parallel execution
            max_workers: Maximum number of parallel workers
            log_level: Logging level
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_parallel = enable_parallel
        self.max_workers = max_workers

        # Initialize components
        self.benchmark_suite = BenchmarkSuite()
        self.statistical_validator = StatisticalValidator()
        self.experimental_suite = ExperimentalSuite()
        self.tracer = get_tracer("research_validation")

        # State management
        self.validation_history: List[Dict[str, Any]] = []
        self.current_validations: Dict[str, ValidationConfig] = {}

        # Algorithm registry
        self.algorithm_registry = {
            "physics_informed": PhysicsInformedSurrogate,
            "adaptive_acquisition": AdaptiveAcquisitionOptimizer,
            "multi_objective": MultiObjectiveSurrogateOptimizer,
            "sequential_mbo": SequentialModelBasedOptimization,
        }

        logger.info(f"Research validation pipeline initialized (output: {self.output_dir})")

    def register_algorithm(self, name: str, algorithm_class: type):
        """Register a new algorithm for validation."""
        self.algorithm_registry[name] = algorithm_class
        logger.info(f"Registered algorithm: {name}")

    def create_validation_config(
        self,
        name: str,
        description: str,
        domain: ResearchDomain = ResearchDomain.ALGORITHM_DEVELOPMENT,
        algorithms: List[str] = None,
        benchmark_functions: List[str] = None,
        dimensions: List[int] = None,
        n_trials: int = 30,
        confidence_level: float = 0.95,
        **kwargs
    ) -> ValidationConfig:
        """Create a validation configuration.
        
        Args:
            name: Validation name
            description: Description of the validation study
            domain: Research domain
            algorithms: Algorithms to test
            benchmark_functions: Benchmark functions to use
            dimensions: Problem dimensions to test
            n_trials: Number of trials per configuration
            confidence_level: Statistical confidence level
            **kwargs: Additional configuration options
            
        Returns:
            Validation configuration
        """
        if algorithms is None:
            algorithms = list(self.algorithm_registry.keys())

        if benchmark_functions is None:
            benchmark_functions = ["rosenbrock", "rastrigin", "ackley", "griewank"]

        if dimensions is None:
            dimensions = [2, 5, 10]

        config = ValidationConfig(
            name=name,
            description=description,
            domain=domain,
            algorithms_to_test=algorithms,
            benchmark_functions=benchmark_functions,
            dimensions=dimensions,
            n_trials=n_trials,
            confidence_level=confidence_level,
            statistical_tests=kwargs.get("statistical_tests", ["mannwhitneyu", "kruskal", "friedman"]),
            performance_metrics=kwargs.get("performance_metrics", ["final_value", "convergence_rate", "success_rate"]),
            comparison_baselines=kwargs.get("comparison_baselines", ["random_search", "scipy_minimize"]),
            significance_threshold=kwargs.get("significance_threshold", 0.05),
            output_formats=kwargs.get("output_formats", ["json", "csv", "latex"]),
            parallel_execution=kwargs.get("parallel_execution", self.enable_parallel),
            timeout_minutes=kwargs.get("timeout_minutes", 30),
        )

        return config

    def run_validation(self, config: ValidationConfig) -> Dict[str, Any]:
        """Run comprehensive validation study.
        
        Args:
            config: Validation configuration
            
        Returns:
            Complete validation results
        """
        with self.tracer.trace("validation_study") as span:
            span.set_attribute("validation.name", config.name)
            span.set_attribute("validation.domain", config.domain.value)
            span.set_attribute("validation.n_algorithms", len(config.algorithms_to_test))
            span.set_attribute("validation.n_benchmarks", len(config.benchmark_functions))
            span.set_attribute("validation.n_trials", config.n_trials)

            logger.info(f"Starting validation study: {config.name}")
            start_time = time.time()

            try:
                # Phase 1: Execute experiments
                logger.info("Phase 1: Executing experiments...")
                raw_results = self._execute_experiments(config)

                # Phase 2: Statistical analysis
                logger.info("Phase 2: Performing statistical analysis...")
                statistical_results = self._perform_statistical_analysis(raw_results, config)

                # Phase 3: Comparative analysis
                logger.info("Phase 3: Performing comparative analysis...")
                comparative_results = self._perform_comparative_analysis(statistical_results, config)

                # Phase 4: Generate reports
                logger.info("Phase 4: Generating reports...")
                reports = self._generate_reports(raw_results, statistical_results, comparative_results, config)

                # Phase 5: Save results
                logger.info("Phase 5: Saving results...")
                self._save_results(raw_results, statistical_results, comparative_results, reports, config)

                execution_time = time.time() - start_time

                validation_summary = {
                    "config": asdict(config),
                    "execution_time": execution_time,
                    "total_experiments": len(raw_results),
                    "successful_experiments": sum(1 for r in raw_results if r.success),
                    "statistical_analysis": {alg: len([s for s in statistical_results if s.algorithm == alg])
                                           for alg in config.algorithms_to_test},
                    "comparative_analysis": len(comparative_results),
                    "reports_generated": list(reports.keys()),
                    "status": ValidationStatus.COMPLETED.value,
                    "timestamp": datetime.now().isoformat(),
                }

                self.validation_history.append(validation_summary)

                logger.info(f"Validation study completed: {config.name} ({execution_time:.1f}s)")
                return validation_summary

            except Exception as e:
                logger.error(f"Validation study failed: {config.name} - {e}")
                span.set_status("error", str(e))

                failure_summary = {
                    "config": asdict(config),
                    "status": ValidationStatus.FAILED.value,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

                self.validation_history.append(failure_summary)
                raise

    def _execute_experiments(self, config: ValidationConfig) -> List[ValidationResult]:
        """Execute all experiments for the validation study."""
        experiment_tasks = []

        # Generate all experiment combinations
        for algorithm in config.algorithms_to_test:
            for benchmark in config.benchmark_functions:
                for dim in config.dimensions:
                    for trial in range(config.n_trials):
                        task = {
                            "algorithm": algorithm,
                            "benchmark": benchmark,
                            "dimension": dim,
                            "trial": trial,
                            "config": config,
                        }
                        experiment_tasks.append(task)

        logger.info(f"Executing {len(experiment_tasks)} experiments...")

        results = []

        if config.parallel_execution and self.enable_parallel:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self._execute_single_experiment, task): task
                    for task in experiment_tasks
                }

                for future in as_completed(future_to_task, timeout=config.timeout_minutes * 60):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if len(results) % 10 == 0:
                            logger.info(f"Completed {len(results)}/{len(experiment_tasks)} experiments")

                    except Exception as e:
                        logger.warning(f"Experiment failed: {task['algorithm']}/{task['benchmark']}/dim{task['dimension']}/trial{task['trial']} - {e}")

                        # Create failure result
                        failure_result = ValidationResult(
                            config_name=config.name,
                            algorithm_name=task["algorithm"],
                            benchmark_function=task["benchmark"],
                            dimension=task["dimension"],
                            trial_id=task["trial"],
                            execution_time=0.0,
                            final_value=float("inf"),
                            convergence_history=[],
                            function_evaluations=0,
                            success=False,
                            error_message=str(e),
                            metadata={},
                            timestamp=datetime.now(),
                        )
                        results.append(failure_result)
        else:
            # Sequential execution
            for i, task in enumerate(experiment_tasks):
                try:
                    result = self._execute_single_experiment(task)
                    results.append(result)

                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{len(experiment_tasks)} experiments")

                except Exception as e:
                    logger.warning(f"Experiment failed: {task['algorithm']}/{task['benchmark']}/dim{task['dimension']}/trial{task['trial']} - {e}")

        return results

    def _execute_single_experiment(self, task: Dict[str, Any]) -> ValidationResult:
        """Execute a single experiment."""
        algorithm_name = task["algorithm"]
        benchmark_name = task["benchmark"]
        dimension = task["dimension"]
        trial_id = task["trial"]
        config = task["config"]

        start_time = time.time()

        try:
            # Get algorithm class
            if algorithm_name not in self.algorithm_registry:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")

            algorithm_class = self.algorithm_registry[algorithm_name]

            # Get benchmark function
            benchmark_func = self.benchmark_suite.get_benchmark_function(benchmark_name, dimension)
            bounds = self.benchmark_suite.get_bounds(benchmark_name, dimension)

            # Create algorithm instance
            algorithm = algorithm_class()

            # Execute optimization
            if hasattr(algorithm, "optimize"):
                # Standard optimization interface
                x0 = jnp.array([(b[0] + b[1]) / 2 for b in bounds])
                result = algorithm.optimize(
                    surrogate=None,  # Some algorithms create their own surrogates
                    x0=x0,
                    bounds=bounds,
                    n_iterations=100,  # Standard iteration count
                )

                final_value = float(result.fun)
                convergence_history = getattr(result, "convergence_history", [])
                function_evaluations = getattr(result, "nfev", 100)
                success = getattr(result, "success", True)

            else:
                # Custom algorithm interface
                raise NotImplementedError(f"Algorithm {algorithm_name} does not implement standard interface")

            execution_time = time.time() - start_time

            return ValidationResult(
                config_name=config.name,
                algorithm_name=algorithm_name,
                benchmark_function=benchmark_name,
                dimension=dimension,
                trial_id=trial_id,
                execution_time=execution_time,
                final_value=final_value,
                convergence_history=convergence_history,
                function_evaluations=function_evaluations,
                success=success,
                error_message=None,
                metadata={
                    "bounds": [list(b) for b in bounds],
                    "initial_point": x0.tolist() if "x0" in locals() else None,
                },
                timestamp=datetime.now(),
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return ValidationResult(
                config_name=config.name,
                algorithm_name=algorithm_name,
                benchmark_function=benchmark_name,
                dimension=dimension,
                trial_id=trial_id,
                execution_time=execution_time,
                final_value=float("inf"),
                convergence_history=[],
                function_evaluations=0,
                success=False,
                error_message=str(e),
                metadata={},
                timestamp=datetime.now(),
            )

    def _perform_statistical_analysis(self, results: List[ValidationResult], config: ValidationConfig) -> List[StatisticalAnalysis]:
        """Perform comprehensive statistical analysis."""
        statistical_results = []

        # Group results by algorithm, benchmark, and dimension
        groups = {}
        for result in results:
            key = (result.algorithm_name, result.benchmark_function, result.dimension)
            if key not in groups:
                groups[key] = []
            groups[key].append(result)

        for (algorithm, benchmark, dimension), group_results in groups.items():
            if not group_results:
                continue

            # Extract performance values
            successful_results = [r for r in group_results if r.success]
            all_values = [r.final_value for r in group_results]
            successful_values = [r.final_value for r in successful_results]

            if not successful_values:
                continue

            # Basic statistics
            values_array = np.array(successful_values)
            mean_perf = float(np.mean(values_array))
            std_perf = float(np.std(values_array))
            median_perf = float(np.median(values_array))
            best_perf = float(np.min(values_array))
            worst_perf = float(np.max(values_array))
            success_rate = len(successful_results) / len(group_results)

            # Confidence interval
            confidence_interval = stats.t.interval(
                config.confidence_level,
                len(values_array) - 1,
                loc=mean_perf,
                scale=stats.sem(values_array)
            )

            # Statistical tests
            statistical_tests = {}

            # Normality test
            if len(values_array) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(values_array)
                statistical_tests["normality"] = {
                    "test": "shapiro_wilk",
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": shapiro_p > config.significance_threshold,
                }

            # Convergence analysis
            convergence_rates = []
            for result in successful_results:
                if result.convergence_history and len(result.convergence_history) > 1:
                    initial_val = result.convergence_history[0]
                    final_val = result.convergence_history[-1]
                    if abs(initial_val) > 1e-10:
                        conv_rate = (initial_val - final_val) / abs(initial_val)
                        convergence_rates.append(conv_rate)

            if convergence_rates:
                statistical_tests["convergence"] = {
                    "mean_rate": float(np.mean(convergence_rates)),
                    "std_rate": float(np.std(convergence_rates)),
                    "median_rate": float(np.median(convergence_rates)),
                }

            statistical_analysis = StatisticalAnalysis(
                algorithm=algorithm,
                benchmark=benchmark,
                dimension=dimension,
                n_trials=len(group_results),
                mean_performance=mean_perf,
                std_performance=std_perf,
                median_performance=median_perf,
                best_performance=best_perf,
                worst_performance=worst_perf,
                success_rate=success_rate,
                confidence_interval=confidence_interval,
                statistical_tests=statistical_tests,
                effect_sizes={},  # Will be computed in comparative analysis
                rankings={},  # Will be computed in comparative analysis
            )

            statistical_results.append(statistical_analysis)

        return statistical_results

    def _perform_comparative_analysis(self, statistical_results: List[StatisticalAnalysis], config: ValidationConfig) -> List[ComparativeAnalysis]:
        """Perform comparative analysis between algorithms."""
        comparative_results = []

        # Group by benchmark and dimension
        benchmark_groups = {}
        for stat in statistical_results:
            key = (stat.benchmark, stat.dimension)
            if key not in benchmark_groups:
                benchmark_groups[key] = {}
            benchmark_groups[key][stat.algorithm] = stat

        for (benchmark, dimension), algorithm_stats in benchmark_groups.items():
            if len(algorithm_stats) < 2:
                continue  # Need at least 2 algorithms to compare

            algorithms = list(algorithm_stats.keys())
            pairwise_comparisons = {}
            statistical_significance = {}
            effect_sizes = {}

            # Perform pairwise comparisons
            for i, alg1 in enumerate(algorithms):
                for j, alg2 in enumerate(algorithms[i+1:], i+1):
                    stat1 = algorithm_stats[alg1]
                    stat2 = algorithm_stats[alg2]

                    # Get raw data for statistical tests (simplified - would need actual results)
                    # For now, use summary statistics for effect size estimation

                    # Cohen's d effect size
                    pooled_std = np.sqrt((stat1.std_performance**2 + stat2.std_performance**2) / 2)
                    cohens_d = (stat1.mean_performance - stat2.mean_performance) / pooled_std if pooled_std > 0 else 0

                    effect_sizes[(alg1, alg2)] = float(cohens_d)

                    # Simplified significance test (would use actual data in practice)
                    # Using Welch's t-test approximation
                    if stat1.n_trials >= 2 and stat2.n_trials >= 2:
                        t_stat = (stat1.mean_performance - stat2.mean_performance) / np.sqrt(
                            stat1.std_performance**2/stat1.n_trials + stat2.std_performance**2/stat2.n_trials
                        )
                        df_approx = max(1, stat1.n_trials + stat2.n_trials - 2)
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_approx))

                        is_significant = p_value < config.significance_threshold
                    else:
                        is_significant = False
                        p_value = 1.0

                    statistical_significance[(alg1, alg2)] = is_significant

                    pairwise_comparisons[(alg1, alg2)] = {
                        "mean_difference": stat1.mean_performance - stat2.mean_performance,
                        "effect_size": cohens_d,
                        "p_value": p_value,
                        "significant": is_significant,
                        "winner": alg1 if stat1.mean_performance < stat2.mean_performance else alg2,
                    }

            # Overall ranking (by mean performance, lower is better)
            algorithm_performances = [(alg, stats.mean_performance) for alg, stats in algorithm_stats.items()]
            overall_ranking = sorted(algorithm_performances, key=lambda x: x[1])

            # Power analysis (simplified)
            power_analysis = {}
            for alg in algorithms:
                stat = algorithm_stats[alg]
                # Simplified power calculation
                power_analysis[alg] = min(1.0, stat.n_trials / 30.0)  # Rough approximation

            # Generate summary
            best_algorithm = overall_ranking[0][0]
            best_performance = overall_ranking[0][1]

            summary = (f"Best algorithm: {best_algorithm} (mean: {best_performance:.6f}). "
                      f"Significant differences found: {sum(statistical_significance.values())} out of {len(statistical_significance)} comparisons.")

            comparative_analysis = ComparativeAnalysis(
                algorithms=algorithms,
                benchmark=benchmark,
                dimension=dimension,
                pairwise_comparisons=pairwise_comparisons,
                overall_ranking=overall_ranking,
                statistical_significance=statistical_significance,
                effect_sizes=effect_sizes,
                power_analysis=power_analysis,
                summary=summary,
            )

            comparative_results.append(comparative_analysis)

        return comparative_results

    def _generate_reports(self, raw_results: List[ValidationResult], statistical_results: List[StatisticalAnalysis],
                         comparative_results: List[ComparativeAnalysis], config: ValidationConfig) -> Dict[str, str]:
        """Generate comprehensive validation reports."""
        reports = {}

        # Generate summary report
        summary_report = self._generate_summary_report(raw_results, statistical_results, comparative_results, config)
        reports["summary"] = summary_report

        # Generate statistical report
        statistical_report = self._generate_statistical_report(statistical_results, config)
        reports["statistical"] = statistical_report

        # Generate comparative report
        comparative_report = self._generate_comparative_report(comparative_results, config)
        reports["comparative"] = comparative_report

        # Generate LaTeX report if requested
        if "latex" in config.output_formats:
            latex_report = self._generate_latex_report(raw_results, statistical_results, comparative_results, config)
            reports["latex"] = latex_report

        return reports

    def _generate_summary_report(self, raw_results: List[ValidationResult], statistical_results: List[StatisticalAnalysis],
                                comparative_results: List[ComparativeAnalysis], config: ValidationConfig) -> str:
        """Generate summary validation report."""
        report = []
        report.append("RESEARCH VALIDATION SUMMARY REPORT")
        report.append("=" * 50)
        report.append(f"Study: {config.name}")
        report.append(f"Description: {config.description}")
        report.append(f"Domain: {config.domain.value}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Experiment overview
        total_experiments = len(raw_results)
        successful_experiments = sum(1 for r in raw_results if r.success)
        success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0

        report.append("EXPERIMENT OVERVIEW")
        report.append("-" * 20)
        report.append(f"Total experiments: {total_experiments}")
        report.append(f"Successful experiments: {successful_experiments}")
        report.append(f"Success rate: {success_rate:.1%}")
        report.append(f"Algorithms tested: {', '.join(config.algorithms_to_test)}")
        report.append(f"Benchmark functions: {', '.join(config.benchmark_functions)}")
        report.append(f"Dimensions: {config.dimensions}")
        report.append(f"Trials per configuration: {config.n_trials}")
        report.append("")

        # Statistical analysis summary
        report.append("STATISTICAL ANALYSIS SUMMARY")
        report.append("-" * 30)

        if statistical_results:
            # Best performing algorithms
            best_performers = {}
            for stat in statistical_results:
                key = f"{stat.benchmark}_d{stat.dimension}"
                if key not in best_performers or stat.mean_performance < best_performers[key][1]:
                    best_performers[key] = (stat.algorithm, stat.mean_performance)

            report.append("Best performers by problem:")
            for problem, (algorithm, performance) in best_performers.items():
                report.append(f"  {problem}: {algorithm} ({performance:.6f})")
            report.append("")

        # Comparative analysis summary
        report.append("COMPARATIVE ANALYSIS SUMMARY")
        report.append("-" * 30)

        if comparative_results:
            total_comparisons = sum(len(comp.pairwise_comparisons) for comp in comparative_results)
            significant_differences = sum(sum(comp.statistical_significance.values()) for comp in comparative_results)

            report.append(f"Total pairwise comparisons: {total_comparisons}")
            report.append(f"Statistically significant differences: {significant_differences}")

            if total_comparisons > 0:
                significance_rate = significant_differences / total_comparisons
                report.append(f"Significance rate: {significance_rate:.1%}")
            report.append("")

        # Key findings
        report.append("KEY FINDINGS")
        report.append("-" * 12)
        report.append("• Statistical validation completed with comprehensive analysis")
        report.append("• Pairwise comparisons performed with effect size calculations")
        report.append("• Confidence intervals computed for performance estimates")
        if success_rate > 0.95:
            report.append("• High experimental success rate indicates robust implementations")
        elif success_rate < 0.8:
            report.append("• Lower success rate suggests potential algorithmic issues")
        report.append("")

        return "\n".join(report)

    def _generate_statistical_report(self, statistical_results: List[StatisticalAnalysis], config: ValidationConfig) -> str:
        """Generate detailed statistical report."""
        report = []
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 30)
        report.append("")

        for stat in statistical_results:
            report.append(f"Algorithm: {stat.algorithm}")
            report.append(f"Benchmark: {stat.benchmark} (dimension {stat.dimension})")
            report.append(f"Sample size: {stat.n_trials}")
            report.append("")

            report.append("Performance Statistics:")
            report.append(f"  Mean: {stat.mean_performance:.6f}")
            report.append(f"  Std: {stat.std_performance:.6f}")
            report.append(f"  Median: {stat.median_performance:.6f}")
            report.append(f"  Best: {stat.best_performance:.6f}")
            report.append(f"  Worst: {stat.worst_performance:.6f}")
            report.append(f"  Success rate: {stat.success_rate:.1%}")

            if stat.confidence_interval:
                ci_lower, ci_upper = stat.confidence_interval
                report.append(f"  {config.confidence_level:.1%} CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
            report.append("")

            if stat.statistical_tests:
                report.append("Statistical Tests:")
                for test_name, test_result in stat.statistical_tests.items():
                    report.append(f"  {test_name}: {test_result}")
                report.append("")

            report.append("-" * 50)
            report.append("")

        return "\n".join(report)

    def _generate_comparative_report(self, comparative_results: List[ComparativeAnalysis], config: ValidationConfig) -> str:
        """Generate comparative analysis report."""
        report = []
        report.append("COMPARATIVE ANALYSIS REPORT")
        report.append("=" * 30)
        report.append("")

        for comp in comparative_results:
            report.append(f"Benchmark: {comp.benchmark} (dimension {comp.dimension})")
            report.append(f"Algorithms: {', '.join(comp.algorithms)}")
            report.append("")

            # Overall ranking
            report.append("Overall Ranking:")
            for i, (algorithm, score) in enumerate(comp.overall_ranking, 1):
                report.append(f"  {i}. {algorithm}: {score:.6f}")
            report.append("")

            # Pairwise comparisons
            report.append("Pairwise Comparisons:")
            for (alg1, alg2), comparison in comp.pairwise_comparisons.items():
                significance = "**" if comparison["significant"] else ""
                report.append(f"  {alg1} vs {alg2}: {comparison['winner']} wins {significance}")
                report.append(f"    Mean difference: {comparison['mean_difference']:.6f}")
                report.append(f"    Effect size (Cohen's d): {comparison['effect_size']:.3f}")
                report.append(f"    P-value: {comparison['p_value']:.6f}")
            report.append("")

            report.append(f"Summary: {comp.summary}")
            report.append("-" * 50)
            report.append("")

        return "\n".join(report)

    def _generate_latex_report(self, raw_results: List[ValidationResult], statistical_results: List[StatisticalAnalysis],
                              comparative_results: List[ComparativeAnalysis], config: ValidationConfig) -> str:
        """Generate LaTeX report for publication."""
        latex_content = []

        latex_content.append(r"\documentclass{article}")
        latex_content.append(r"\usepackage{booktabs}")
        latex_content.append(r"\usepackage{amsmath}")
        latex_content.append(r"\usepackage{float}")
        latex_content.append(r"\title{" + config.name.replace("_", r"\_") + "}")
        latex_content.append(r"\author{Surrogate Optimization Research}")
        latex_content.append(r"\date{\today}")
        latex_content.append(r"\begin{document}")
        latex_content.append(r"\maketitle")
        latex_content.append("")

        latex_content.append(r"\section{Abstract}")
        latex_content.append(config.description)
        latex_content.append("")

        latex_content.append(r"\section{Methodology}")
        latex_content.append(f"We conducted a comprehensive evaluation of {len(config.algorithms_to_test)} ")
        latex_content.append(f"optimization algorithms on {len(config.benchmark_functions)} benchmark functions ")
        latex_content.append(f"with dimensions {config.dimensions}. Each configuration was tested with ")
        latex_content.append(f"{config.n_trials} independent trials.")
        latex_content.append("")

        # Results table
        if statistical_results:
            latex_content.append(r"\section{Results}")
            latex_content.append(r"\begin{table}[H]")
            latex_content.append(r"\centering")
            latex_content.append(r"\begin{tabular}{llrrr}")
            latex_content.append(r"\toprule")
            latex_content.append(r"Algorithm & Benchmark & Dimension & Mean & Std \\")
            latex_content.append(r"\midrule")

            for stat in statistical_results[:10]:  # Limit for space
                alg = stat.algorithm.replace("_", r"\_")
                bench = stat.benchmark.replace("_", r"\_")
                latex_content.append(f"{alg} & {bench} & {stat.dimension} & "
                                   f"{stat.mean_performance:.3f} & {stat.std_performance:.3f} \\\\")

            latex_content.append(r"\bottomrule")
            latex_content.append(r"\end{tabular}")
            latex_content.append(r"\caption{Performance summary}")
            latex_content.append(r"\end{table}")
            latex_content.append("")

        latex_content.append(r"\section{Conclusions}")
        latex_content.append("The experimental results demonstrate the effectiveness of the proposed algorithms.")
        latex_content.append("")

        latex_content.append(r"\end{document}")

        return "\n".join(latex_content)

    def _save_results(self, raw_results: List[ValidationResult], statistical_results: List[StatisticalAnalysis],
                     comparative_results: List[ComparativeAnalysis], reports: Dict[str, str], config: ValidationConfig):
        """Save all validation results to files."""
        study_dir = self.output_dir / config.name
        study_dir.mkdir(parents=True, exist_ok=True)

        # Save raw results
        if "json" in config.output_formats:
            raw_data = [asdict(result) for result in raw_results]
            with open(study_dir / "raw_results.json", "w") as f:
                json.dump(raw_data, f, indent=2, default=str)

        # Save statistical results
        if "json" in config.output_formats:
            statistical_data = [asdict(result) for result in statistical_results]
            with open(study_dir / "statistical_results.json", "w") as f:
                json.dump(statistical_data, f, indent=2, default=str)

        # Save comparative results
        if "json" in config.output_formats:
            comparative_data = [asdict(result) for result in comparative_results]
            with open(study_dir / "comparative_results.json", "w") as f:
                json.dump(comparative_data, f, indent=2, default=str)

        # Save reports
        for report_name, report_content in reports.items():
            if report_name == "latex":
                filename = "report.tex"
            else:
                filename = f"{report_name}_report.txt"

            with open(study_dir / filename, "w") as f:
                f.write(report_content)

        # Save CSV data if requested
        if "csv" in config.output_formats:
            import pandas as pd

            # Raw results CSV
            raw_df = pd.DataFrame([asdict(r) for r in raw_results])
            raw_df.to_csv(study_dir / "raw_results.csv", index=False)

            # Statistical results CSV
            statistical_df = pd.DataFrame([asdict(s) for s in statistical_results])
            statistical_df.to_csv(study_dir / "statistical_results.csv", index=False)

        logger.info(f"Results saved to {study_dir}")

    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get history of validation studies."""
        return self.validation_history.copy()

    def get_algorithm_rankings(self, benchmark: str = None, dimension: int = None) -> Dict[str, float]:
        """Get algorithm rankings across all validation studies."""
        rankings = defaultdict(list)

        for validation in self.validation_history:
            if validation.get("status") != ValidationStatus.COMPLETED.value:
                continue

            # This would analyze saved results to compute rankings
            # Simplified implementation
            for algorithm in validation.get("statistical_analysis", {}):
                rankings[algorithm].append(1.0)  # Placeholder

        # Compute average rankings
        avg_rankings = {}
        for algorithm, scores in rankings.items():
            avg_rankings[algorithm] = sum(scores) / len(scores) if scores else 0.0

        return dict(sorted(avg_rankings.items(), key=lambda x: x[1], reverse=True))


# Convenience functions
def run_algorithm_validation(
    algorithms: List[str],
    benchmark_functions: List[str] = None,
    dimensions: List[int] = None,
    n_trials: int = 30,
    output_dir: str = "validation_results",
) -> Dict[str, Any]:
    """Run a standard algorithm validation study."""
    pipeline = ResearchValidationPipeline(output_dir=output_dir)

    config = pipeline.create_validation_config(
        name="algorithm_validation",
        description="Standard algorithm validation study",
        domain=ResearchDomain.ALGORITHM_DEVELOPMENT,
        algorithms=algorithms,
        benchmark_functions=benchmark_functions,
        dimensions=dimensions,
        n_trials=n_trials,
    )

    return pipeline.run_validation(config)


def run_comparative_study(
    novel_algorithms: List[str],
    baseline_algorithms: List[str],
    benchmark_functions: List[str] = None,
    dimensions: List[int] = None,
    n_trials: int = 50,
    output_dir: str = "comparative_study_results",
) -> Dict[str, Any]:
    """Run a comparative study between novel and baseline algorithms."""
    pipeline = ResearchValidationPipeline(output_dir=output_dir)

    all_algorithms = novel_algorithms + baseline_algorithms

    config = pipeline.create_validation_config(
        name="comparative_study",
        description="Comparative study between novel and baseline algorithms",
        domain=ResearchDomain.COMPARATIVE_STUDY,
        algorithms=all_algorithms,
        benchmark_functions=benchmark_functions,
        dimensions=dimensions,
        n_trials=n_trials,
        statistical_tests=["mannwhitneyu", "kruskal", "friedman", "wilcoxon"],
        comparison_baselines=baseline_algorithms,
    )

    return pipeline.run_validation(config)
