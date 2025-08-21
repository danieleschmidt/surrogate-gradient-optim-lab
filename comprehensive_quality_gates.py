#!/usr/bin/env python3
"""
Comprehensive Quality Gates Runner
Tests security, performance, and production readiness
"""

import sys
import time
import subprocess
import json
from pathlib import Path
import importlib.util
import traceback

def run_security_checks():
    """Run security validation checks."""
    print("🔐 SECURITY QUALITY GATE")
    print("=" * 40)
    
    security_results = {}
    
    # 1. Import Safety Check
    print("🔍 Checking for unsafe imports...")
    unsafe_patterns = [
        "eval(",
        "exec(",
        "subprocess.call",
        "os.system",
        "__import__",
        "pickle.loads"  # Can be unsafe with untrusted data
    ]
    
    python_files = list(Path("/root/repo").glob("**/*.py"))
    security_issues = []
    
    for file_path in python_files:
        if "/.git/" in str(file_path) or "__pycache__" in str(file_path):
            continue
            
        try:
            content = file_path.read_text()
            for pattern in unsafe_patterns:
                if pattern in content:
                    security_issues.append(f"{file_path}: {pattern}")
        except Exception:
            pass
    
    if security_issues:
        print("⚠️  Security issues found:")
        for issue in security_issues[:5]:  # Show first 5
            print(f"  - {issue}")
        security_results["unsafe_imports"] = False
    else:
        print("✅ No unsafe imports detected")
        security_results["unsafe_imports"] = True
    
    # 2. Hardcoded Secrets Check
    print("\n🔍 Checking for hardcoded secrets...")
    secret_patterns = [
        "password",
        "api_key",
        "secret_key",
        "token",
        "auth_token"
    ]
    
    secret_issues = []
    for file_path in python_files:
        if "/.git/" in str(file_path) or "__pycache__" in str(file_path):
            continue
            
        try:
            content = file_path.read_text().lower()
            for pattern in secret_patterns:
                if f"{pattern} =" in content or f'"{pattern}"' in content:
                    secret_issues.append(f"{file_path}: {pattern}")
        except Exception:
            pass
    
    if secret_issues:
        print("⚠️  Potential hardcoded secrets:")
        for issue in secret_issues[:3]:
            print(f"  - {issue}")
        security_results["hardcoded_secrets"] = False
    else:
        print("✅ No hardcoded secrets detected")
        security_results["hardcoded_secrets"] = True
    
    # 3. Input Validation Check
    print("\n🔍 Checking input validation patterns...")
    has_validation = False
    
    for file_path in python_files:
        if "quality_gates" in str(file_path):
            continue
            
        try:
            content = file_path.read_text()
            if "validate" in content.lower() or "isfinite" in content or "ValidationError" in content:
                has_validation = True
                break
        except Exception:
            pass
    
    if has_validation:
        print("✅ Input validation patterns found")
        security_results["input_validation"] = True
    else:
        print("⚠️  Limited input validation detected")
        security_results["input_validation"] = False
    
    return security_results

def run_performance_tests():
    """Run performance benchmarks."""
    print("\n⚡ PERFORMANCE QUALITY GATE")
    print("=" * 40)
    
    performance_results = {}
    
    # Test basic performance
    print("🔍 Testing basic prediction performance...")
    
    try:
        import jax.numpy as jnp
        import sys
        sys.path.append('/root/repo')
        
        # Import our simple surrogate
        from simple_surrogate_gen1 import SimpleSurrogate, collect_random_data
        
        # Simple performance test
        def test_function(x):
            return float(jnp.sum(x**2))
        
        bounds = [(-2, 2), (-2, 2)]
        
        # Data collection performance
        start_time = time.time()
        X, y = collect_random_data(test_function, 100, bounds)
        data_time = time.time() - start_time
        data_throughput = len(X) / data_time if data_time > 0 else 0
        
        print(f"  📊 Data collection: {len(X)} samples in {data_time:.3f}s ({data_throughput:.1f} samples/sec)")
        
        # Training performance
        surrogate = SimpleSurrogate()
        start_time = time.time()
        surrogate.fit(X, y)
        train_time = time.time() - start_time
        train_throughput = len(X) / train_time if train_time > 0 else 0
        
        print(f"  🧠 Training: {train_time:.3f}s ({train_throughput:.1f} samples/sec)")
        
        # Prediction performance
        test_points = [jnp.array([0.5, 0.5]), jnp.array([1.0, 1.0]), jnp.array([-1.0, -1.0])]
        
        start_time = time.time()
        predictions = [surrogate.predict(x) for x in test_points * 100]  # 300 predictions
        pred_time = time.time() - start_time
        pred_throughput = len(predictions) / pred_time if pred_time > 0 else 0
        
        print(f"  🔮 Prediction: {len(predictions)} predictions in {pred_time:.3f}s ({pred_throughput:.1f} pred/sec)")
        
        performance_results = {
            "data_throughput": data_throughput,
            "train_throughput": train_throughput,
            "prediction_throughput": pred_throughput,
            "data_time": data_time,
            "train_time": train_time,
            "prediction_time": pred_time
        }
        
        # Performance criteria
        criteria = {
            "Fast data collection": data_throughput > 20,
            "Fast training": train_throughput > 50,
            "Fast prediction": pred_throughput > 100,
            "Reasonable total time": (data_time + train_time) < 5.0
        }
        
        print(f"\n  📈 Performance Criteria:")
        all_passed = True
        for criterion, passed in criteria.items():
            status = "✅ PASS" if passed else "⚠️  SLOW"
            print(f"    {status}: {criterion}")
            all_passed = all_passed and passed
        
        performance_results["performance_acceptable"] = all_passed
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        performance_results["performance_acceptable"] = False
        traceback.print_exc()
    
    return performance_results

def run_functionality_tests():
    """Test core functionality."""
    print("\n🔧 FUNCTIONALITY QUALITY GATE")
    print("=" * 40)
    
    functionality_results = {}
    
    # Test basic imports
    print("🔍 Testing core imports...")
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import scipy
        print("✅ Core dependencies available")
        functionality_results["imports"] = True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        functionality_results["imports"] = False
    
    # Test JAX functionality
    print("\n🔍 Testing JAX functionality...")
    try:
        import jax.numpy as jnp
        
        # Basic array operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x**2)
        
        # Gradient computation
        def f(x):
            return jnp.sum(x**2)
        
        grad_f = jax.grad(f)
        gradient = grad_f(x)
        
        print("✅ JAX operations working")
        functionality_results["jax_operations"] = True
        
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
        functionality_results["jax_operations"] = False
    
    # Test optimization capability
    print("\n🔍 Testing optimization capability...")
    try:
        # Simple optimization test
        def simple_objective(x):
            return float(jnp.sum((x - 1.0)**2))
        
        # Simple gradient descent
        x = jnp.array([0.0, 0.0])
        for _ in range(10):
            grad = jax.grad(simple_objective)(x)
            x = x - 0.1 * grad
        
        final_value = simple_objective(x)
        
        if final_value < 0.1:  # Should be close to optimum
            print("✅ Basic optimization working")
            functionality_results["optimization"] = True
        else:
            print(f"⚠️  Optimization not converging well: final value {final_value}")
            functionality_results["optimization"] = False
        
    except Exception as e:
        print(f"❌ Optimization test failed: {e}")
        functionality_results["optimization"] = False
    
    return functionality_results

def run_code_quality_checks():
    """Run code quality checks."""
    print("\n📋 CODE QUALITY GATE")
    print("=" * 40)
    
    quality_results = {}
    
    # Check for basic code structure
    print("🔍 Checking code structure...")
    
    required_files = [
        "simple_surrogate_gen1.py",
        "robust_surrogate_gen2.py", 
        "scalable_surrogate_gen3.py"
    ]
    
    missing_files = []
    for filename in required_files:
        if not Path(f"/root/repo/{filename}").exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        quality_results["required_files"] = False
    else:
        print("✅ All required files present")
        quality_results["required_files"] = True
    
    # Check for documentation
    print("\n🔍 Checking documentation...")
    
    has_readme = Path("/root/repo/README.md").exists()
    has_docstrings = False
    
    # Check for docstrings in Python files
    for file_path in Path("/root/repo").glob("*.py"):
        try:
            content = file_path.read_text()
            if '"""' in content and ("Args:" in content or "Returns:" in content):
                has_docstrings = True
                break
        except Exception:
            pass
    
    print(f"✅ README.md: {'present' if has_readme else 'missing'}")
    print(f"✅ Docstrings: {'found' if has_docstrings else 'limited'}")
    
    quality_results["documentation"] = has_readme and has_docstrings
    
    # Check for error handling
    print("\n🔍 Checking error handling...")
    
    has_error_handling = False
    for file_path in Path("/root/repo").glob("*.py"):
        if "quality_gates" in str(file_path):
            continue
            
        try:
            content = file_path.read_text()
            if "try:" in content and "except" in content:
                has_error_handling = True
                break
        except Exception:
            pass
    
    if has_error_handling:
        print("✅ Error handling patterns found")
        quality_results["error_handling"] = True
    else:
        print("⚠️  Limited error handling")
        quality_results["error_handling"] = False
    
    return quality_results

def main():
    """Run comprehensive quality gates."""
    print("🛡️ COMPREHENSIVE QUALITY GATES")
    print("=" * 60)
    
    all_results = {}
    
    # Run all quality gate categories
    all_results["security"] = run_security_checks()
    all_results["performance"] = run_performance_tests()  
    all_results["functionality"] = run_functionality_tests()
    all_results["code_quality"] = run_code_quality_checks()
    
    # Overall assessment
    print("\n🏆 OVERALL QUALITY GATE RESULTS")
    print("=" * 60)
    
    category_results = {}
    
    for category, results in all_results.items():
        passed_checks = sum(1 for v in results.values() if v is True)
        total_checks = len(results)
        pass_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        category_results[category] = {
            "passed": passed_checks,
            "total": total_checks,
            "pass_rate": pass_rate,
            "status": "PASS" if pass_rate >= 0.7 else "NEEDS_WORK"
        }
        
        print(f"{category.upper()}: {passed_checks}/{total_checks} checks passed "
              f"({pass_rate:.1%}) - {category_results[category]['status']}")
    
    # Final verdict
    overall_pass_rate = sum(r["passed"] for r in category_results.values()) / sum(r["total"] for r in category_results.values())
    overall_status = "PASS" if overall_pass_rate >= 0.7 else "NEEDS_IMPROVEMENT"
    
    print(f"\nOVERALL: {overall_pass_rate:.1%} checks passed - {overall_status}")
    
    if overall_status == "PASS":
        print("\n✅ QUALITY GATES PASSED - Ready for deployment consideration")
    else:
        print("\n⚠️  QUALITY GATES NEED WORK - Address issues before deployment")
    
    # Save results
    results_file = Path("/root/repo/quality_gates_report.json")
    with open(results_file, "w") as f:
        json.dump({
            "overall_status": overall_status,
            "overall_pass_rate": overall_pass_rate,
            "categories": category_results,
            "detailed_results": all_results,
            "timestamp": time.time()
        }, f, indent=2)
    
    print(f"\n📝 Detailed results saved to: {results_file}")
    
    return overall_status == "PASS"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)