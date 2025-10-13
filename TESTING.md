# Testing Guide

Complete guide to testing the temporal AME package.

## Installation

First, install testing dependencies:

```bash
pip install pytest pytest-cov
```

Or install all dependencies including tests:

```bash
pip install -r requirements.txt
```

## Quick Start

### Run All Tests

```bash
# From project root
pytest

# Or use the provided script
chmod +x run_tests.sh
./run_tests.sh
```

### Expected Output

```
========================================
Temporal AME Test Suite
========================================

collected 150 items

tests/test_models.py ...................... [ 20%]
tests/test_inference.py ..................... [ 50%]
tests/test_utils.py ......................... [ 75%]
tests/test_visualization.py ................ [100%]

========================================
All tests passed! ✓
========================================

150 passed in 45.23s
```

## Test Coverage

Our test suite covers:

### Models (`test_models.py`)
- ✓ Static AME model initialization
- ✓ Temporal AME model initialization
- ✓ Data generation (static and temporal)
- ✓ Mean structure computation
- ✓ Reconstruction error calculation
- ✓ Contribution analysis
- ✓ AR(1) dynamics verification
- ✓ Reproducibility with seeds

### Inference (`test_inference.py`)
- ✓ Naive MF initialization and fitting
- ✓ Structured MF (good/bad factorization)
- ✓ ELBO computation and convergence
- ✓ Variational parameter retrieval
- ✓ Forward prediction
- ✓ Different learning rates
- ✓ Method comparison
- ✓ Structure preservation during optimization

### Utilities (`test_utils.py`)
- ✓ Reconstruction error computation
- ✓ Contribution analysis
- ✓ Temporal contributions
- ✓ Procrustes alignment
- ✓ Sign alignment
- ✓ Temporal state alignment
- ✓ All metrics (MSE, RMSE, MAE, R², correlation, etc.)
- ✓ Link prediction metrics
- ✓ Calibration and coverage
- ✓ Diagnostic summaries

### Visualization (`test_visualization.py`)
- ✓ All static plots (convergence, network, latent space)
- ✓ All temporal plots (trajectories, contributions)
- ✓ All comparison plots (methods, recovery)
- ✓ Plot saving functionality
- ✓ Edge cases (empty data, single node, etc.)

## Test Statistics

| Category      | Test Count | Coverage |
|---------------|------------|----------|
| Models        | 35+        | >90%     |
| Inference     | 40+        | >85%     |
| Utils         | 45+        | >90%     |
| Visualization | 30+        | >80%     |
| **Total**     | **150+**   | **>85%** |

## Advanced Usage

### Run Specific Tests

```bash
# Test only models
pytest tests/test_models.py

# Test specific class
pytest tests/test_models.py::TestTemporalAMEModel

# Test specific function
pytest tests/test_models.py::TestTemporalAMEModel::test_initialization
```

### Verbose Mode

```bash
# Show detailed output
pytest -v

# Show print statements
pytest -s

# Both
pytest -vs
```

### Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View the report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Fast Tests Only

```bash
# Skip slow tests
pytest -m "not slow"

# Or use the script
./run_tests.sh --fast
```

### Stop on First Failure

```bash
# Useful for debugging
pytest -x
```

### Show Test Durations

```bash
# Show 10 slowest tests
pytest --durations=10
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Test Development Workflow

### 1. Write Test First (TDD)

```python
# tests/test_new_feature.py
def test_new_feature():
    """Test the new feature."""
    result = new_feature(input_data)
    assert result == expected_output
```

### 2. Run Test (Should Fail)

```bash
pytest tests/test_new_feature.py
```

### 3. Implement Feature

```python
# src/module.py
def new_feature(input_data):
    # Implementation
    return output
```

### 4. Run Test (Should Pass)

```bash
pytest tests/test_new_feature.py
```

### 5. Refactor if Needed

While keeping tests passing.

## Common Issues

### Issue: Import Errors

**Problem:**
```
ImportError: No module named 'src'
```

**Solution:**
```bash
# Run from project root
cd /path/to/temporal-ame

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Tests Pass Locally but Fail in CI

**Problem:** Random test failures

**Solution:**
- Check random seeds are set
- Verify fixture isolation
- Look for file system dependencies

### Issue: Slow Test Suite

**Problem:** Tests take too long

**Solutions:**
```bash
# Run in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto

# Skip slow tests during development
pytest -m "not slow"

# Profile to find slow tests
pytest --durations=0
```

### Issue: Coverage Not Updating

**Problem:** New code not reflected in coverage

**Solution:**
```bash
# Clear cache
pytest --cache-clear

# Regenerate coverage
rm -rf .coverage htmlcov/
pytest --cov=src --cov-report=html
```

## Best Practices

1. **Run tests before committing**
   ```bash
   pytest
   ```

2. **Write tests for new features**
   - Aim for >80% coverage
   - Test both success and failure cases

3. **Keep tests fast**
   - Use small test data
   - Mock expensive operations
   - Mark slow tests with `@pytest.mark.slow`

4. **Use descriptive test names**
   ```python
   # Good
   def test_model_raises_error_on_negative_nodes():
       pass
   
   # Bad
   def test_error():
       pass
   ```

5. **Test edge cases**
   - Empty inputs
   - Single element
   - Maximum values
   - Invalid inputs

6. **Use fixtures for common setup**
   ```python
   @pytest.fixture
   def trained_model():
       model = Model()
       model.train(data)
       return model
   ```

## Debugging Tests

### Print Debug Information

```bash
# Show print statements
pytest -s

# Show local variables on failure
pytest -l
```

### Use Debugger

```python
def test_something():
    result = compute_something()
    import pdb; pdb.set_trace()  # Breakpoint
    assert result == expected
```

### Verbose Failure Output

```bash
# Show full diff on assertion failure
pytest -vv
```

## Maintenance

### Update Tests When Code Changes

- Add tests for new features
- Update tests for modified behavior
- Remove tests for deprecated features
- Keep test documentation current

### Regular Coverage Checks

```bash
# Check coverage regularly
pytest --cov=src --cov-report=term-missing
```

Target: >80% overall coverage

## Resources

- **Pytest Docs**: https://docs.pytest.org/
- **Test Coverage**: https://coverage.readthedocs.io/
- **Testing Best Practices**: https://docs.python-guide.org/writing/tests/
- **PyTorch Testing**: https://pytorch.org/docs/stable/testing.html

## Questions?

If tests fail unexpectedly:
1. Check the error message carefully
2. Run with `-v` for more details
3. Check `tests/README.md` for troubleshooting
4. Open an issue with full error output