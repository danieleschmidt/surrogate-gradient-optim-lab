# ADR-0001: JAX Backend Selection for Automatic Differentiation

## Status
Accepted

## Context
We need to choose a backend for automatic differentiation and numerical computing in our surrogate gradient optimization library. The core functionality relies heavily on gradient computation, which makes this choice critical for performance and usability.

## Decision
We will use JAX as the primary backend for automatic differentiation and numerical computing.

## Rationale

### Advantages of JAX:
1. **Automatic Differentiation**: Native support for forward and reverse-mode AD
2. **JIT Compilation**: XLA compilation for performance optimization
3. **GPU/TPU Support**: Seamless acceleration without code changes
4. **Functional Programming**: Pure functions enable better testing and reproducibility
5. **NumPy Compatibility**: Familiar API with additional capabilities
6. **Ecosystem**: Growing ecosystem with Optax, Flax, and other ML libraries

### Alternatives Considered:
- **PyTorch**: Excellent AD but more focused on neural networks
- **TensorFlow**: More complex API, less suitable for numerical optimization
- **Autograd**: Simpler but less performance optimization
- **Pure NumPy**: No automatic differentiation capabilities

### Trade-offs:
- **Learning Curve**: JAX has functional programming paradigm
- **Ecosystem Maturity**: Smaller ecosystem compared to PyTorch/TensorFlow
- **Memory Usage**: JIT compilation can increase memory usage

## Consequences

### Positive:
- Fast gradient computation for surrogate models
- GPU acceleration for large-scale optimization
- Clean, functional API for optimization algorithms
- Easy integration with existing numerical Python ecosystem

### Negative:
- Additional dependency with specific version requirements
- Learning curve for contributors familiar with imperative frameworks
- Potential debugging complexity with JIT compilation

## Implementation Details
- Use `jax.numpy` for array operations
- Leverage `jax.grad` for gradient computation
- Apply `jax.jit` for performance-critical functions
- Use `jax.vmap` for batch operations

## Monitoring
- Track performance benchmarks against NumPy baseline
- Monitor memory usage patterns with JIT compilation
- Collect user feedback on API usability

## Date
2025-01-15