#!/usr/bin/env python3
"""Simple Generation 1 functionality verification."""

import sys
import numpy as np

# Add project to path
sys.path.insert(0, '/root/repo')

def test_core_imports():
    """Test basic imports work."""
    try:
        from surrogate_optim.core import SurrogateOptimizer, quick_optimize
        from surrogate_optim.models.base import Dataset
        from surrogate_optim.data.collector import collect_data
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False

def test_basic_optimization():
    """Test basic optimization pipeline."""
    try:
        import jax.numpy as jnp
        from surrogate_optim.core import SurrogateOptimizer
        from surrogate_optim.models.base import Dataset
        
        # Simple test function
        def simple_func(x):
            return float(jnp.sum(x**2))
        
        # Generate test data
        X = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        y = jnp.array([simple_func(x) for x in X])
        dataset = Dataset(X=X, y=y)
        
        # Create and train optimizer
        optimizer = SurrogateOptimizer(surrogate_type="neural_network")
        optimizer.fit_surrogate(dataset)
        
        # Test prediction
        test_point = jnp.array([0.2, 0.3])
        prediction = optimizer.predict(test_point)
        
        print(f"✅ Basic optimization working - prediction: {float(prediction):.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Basic optimization failed: {e}")
        return False

def test_research_imports():
    """Test research algorithm imports."""
    try:
        from surrogate_optim.research.novel_algorithms import (
            PhysicsInformedSurrogate, AdaptiveAcquisitionOptimizer
        )
        from surrogate_optim.research.experimental_suite import ResearchExperimentSuite
        print("✅ Research algorithm imports successful")
        return True
    except ImportError as e:
        print(f"❌ Research import failed: {e}")
        return False

def main():
    """Run simple Generation 1 tests."""
    print("=" * 50)
    print("🧪 GENERATION 1: SIMPLE FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        test_core_imports,
        test_basic_optimization,
        test_research_imports,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Generation 1: MAKE IT WORK - SUCCESSFUL!")
        return True
    else:
        print("⚠️  Some basic functionality needs work")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)