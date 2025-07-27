# ADR-0002: Surrogate Model Architecture Design

## Status
Accepted

## Context
We need to design a flexible architecture for surrogate models that supports multiple model types (neural networks, Gaussian processes, random forests) while providing a consistent interface for optimization algorithms.

## Decision
We will implement a plugin-based architecture with a common `Surrogate` abstract base class and specialized implementations for each model type.

## Rationale

### Architecture Benefits:
1. **Extensibility**: Easy to add new surrogate model types
2. **Consistency**: Uniform interface for all optimization algorithms
3. **Flexibility**: Support for hybrid and ensemble models
4. **Testing**: Each model type can be tested independently
5. **Performance**: Specialized implementations for optimal performance

### Design Principles:
- **Interface Segregation**: Minimal required methods in base class
- **Single Responsibility**: Each model focuses on its specific approach
- **Open/Closed**: Open for extension, closed for modification
- **Dependency Inversion**: Algorithms depend on abstractions, not implementations

## Interface Design

```python
class Surrogate(ABC):
    @abstractmethod
    def predict(self, x: Array) -> float:
        """Predict function value at point x"""
        pass
    
    @abstractmethod
    def gradient(self, x: Array) -> Array:
        """Compute gradient at point x"""
        pass
    
    def uncertainty(self, x: Array) -> float:
        """Optional: Return prediction uncertainty"""
        return 0.0
    
    def batch_predict(self, X: Array) -> Array:
        """Batch prediction with default vectorization"""
        return jax.vmap(self.predict)(X)
```

## Model-Specific Implementations

### Neural Network Surrogates
- Use JAX neural network libraries (Flax/Haiku)
- Support ensemble methods for uncertainty estimation
- Enable gradient matching loss functions

### Gaussian Process Surrogates
- Implement analytical gradients from GP posterior
- Use GPyTorch or custom JAX implementation
- Support automatic kernel selection

### Random Forest Surrogates
- Implement gradient smoothing techniques
- Use scikit-learn with JAX wrapper for gradients
- Support feature importance analysis

## Consequences

### Positive:
- Clear separation of concerns between model types
- Easy to benchmark different surrogate approaches
- Consistent API for optimization algorithms
- Support for future model extensions

### Negative:
- Initial overhead in implementing abstract interfaces
- Potential performance cost of abstraction layer
- Need to maintain consistency across implementations

## Implementation Guidelines
- Use factory pattern for model instantiation
- Implement comprehensive unit tests for each model type
- Provide configuration validation for model parameters
- Document performance characteristics of each model

## Future Considerations
- Support for composite/hybrid models
- Automatic model selection based on data characteristics
- Integration with AutoML frameworks
- Support for custom user-defined models

## Date
2025-01-15