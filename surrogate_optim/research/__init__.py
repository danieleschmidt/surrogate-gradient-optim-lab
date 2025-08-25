"""Research module for novel algorithms and experimental features."""

from .experimental_suite import (
    AblationStudy,
    ComparisonStudy,
    ResearchExperimentSuite,
    run_research_experiments,
)
from .novel_algorithms import (
    AdaptiveAcquisitionOptimizer,
    MultiObjectiveSurrogateOptimizer,
    PhysicsInformedSurrogate,
    ResearchResult,
    SequentialModelBasedOptimization,
)
from .theoretical_analysis import (
    ConvergenceAnalyzer,
    GradientErrorAnalyzer,
    SampleComplexityAnalyzer,
    TheoreticalBounds,
)
from .validation_pipeline import (
    ComparativeAnalysis,
    ResearchDomain,
    ResearchValidationPipeline,
    StatisticalAnalysis,
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
    run_algorithm_validation,
    run_comparative_study,
)

__all__ = [
    # Novel algorithms
    "PhysicsInformedSurrogate",
    "AdaptiveAcquisitionOptimizer",
    "MultiObjectiveSurrogateOptimizer",
    "SequentialModelBasedOptimization",
    "ResearchResult",

    # Theoretical analysis
    "ConvergenceAnalyzer",
    "SampleComplexityAnalyzer",
    "GradientErrorAnalyzer",
    "TheoreticalBounds",

    # Experimental framework
    "ResearchExperimentSuite",
    "ComparisonStudy",
    "AblationStudy",
    "run_research_experiments",

    # Validation pipeline
    "ValidationStatus",
    "ResearchDomain",
    "ValidationConfig",
    "ValidationResult",
    "StatisticalAnalysis",
    "ComparativeAnalysis",
    "ResearchValidationPipeline",
    "run_algorithm_validation",
    "run_comparative_study",
]
