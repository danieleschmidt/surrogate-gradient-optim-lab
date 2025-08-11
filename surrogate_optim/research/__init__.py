"""Research module for novel algorithms and experimental features."""

from .novel_algorithms import (
    PhysicsInformedSurrogate,
    AdaptiveAcquisitionOptimizer,
    MultiObjectiveSurrogateOptimizer,
    SequentialModelBasedOptimization,
    ResearchResult,
)

from .theoretical_analysis import (
    ConvergenceAnalyzer,
    SampleComplexityAnalyzer,
    GradientErrorAnalyzer,
    TheoreticalBounds,
)

from .experimental_suite import (
    ResearchExperimentSuite,
    ComparisonStudy,
    AblationStudy,
    run_research_experiments,
)

from .validation_pipeline import (
    ValidationStatus,
    ResearchDomain,
    ValidationConfig,
    ValidationResult,
    StatisticalAnalysis,
    ComparativeAnalysis,
    ResearchValidationPipeline,
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