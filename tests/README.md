# Test Suite

Comprehensive test suite for the temporal AME package using pytest.

## Structure

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Shared fixtures
├── test_models.py           # Model tests
├── test_inference.py        # Inference method tests
├── test_utils.py            # Utility function tests
├── test_visualization.py    # Visualization tests
└── README.md                # This file
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Or use the provided script
./run_tests.sh
```

### Run Specific Test Files

```bash
# Test only models
pytest tests/test_models.py

# Test only inference methods
pytest tests/test_inference.py

# Test specific class
pytest tests/test_models.py::TestStaticAMEModel

# Test specific function
pytest tests/test_models.py::TestStaticAMEModel::test_initialization
```

### Run with Options

```bash
# Verbose output
pytest -v

# Stop at first failure
pytest -x

# Show print statements
pytest -s

# Run in parallel (requires pytest-xdist)
pytest -n auto

# Skip slow tests
pytest -m "not slow"

# Only visualization tests
pytest -m visualization
```

### Coverage Reports

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

### Using the Test Script

```bash
# Run all tests
./run_tests.sh

# Skip slow tests
./run_tests.sh --fast

# With coverage
./run_tests.sh --coverage

# Specific test file
./run_tests.sh --test test_models.py

# Help
./run_tests.sh --help
```

## Test Organization

### Unit Tests

Test individual functions and methods in isolation:
- `test_models.py`: Model initialization, data generation, mean computation
- `test_inference.py`: VI initialization, fitting, convergence
- `test_utils.py`: Diagnostic, alignment, and metric functions

### Integration Tests

Test interactions between components:
- Inference methods working with models
- End-to-end workflows
- Method comparisons

### Visualization Tests

Test that plotting functions run without errors:
- Check figure objects are created
- Verify axes and subplots
- Test saving to files
- Use non-interactive backend (Agg)

## Fixtures

Shared fixtures in `conftest.py`:

- `seed`: Standard random seed (42)
- `small_network_params`: Parameters for small test networks
- `temporal_network_params`: Parameters for temporal networks
- `static_model`: Pre-initialized static AME model
- `temporal_model`: Pre-initialized temporal AME model
- `static_data`: Generated static network data with latents
- `temporal_data`: Generated temporal network data with latents
- `mock_history`: Mock optimization history
- `sample_trajectories`: Sample true/estimated trajectories

## Writing New Tests

### Template

```python
import pytest
import torch
from src.models import YourModel

class TestYourFeature:
    """Tests for your feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        model = YourModel(param=value)
        
        # Act
        result = model.do_something()
        
        # Assert
        assert result is not None
        assert result.shape == expected_shape
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            model = YourModel(invalid_param=-1)
```

### Best Practices

1. **Use descriptive names**: `test_model_generates_correct_shape` not `test_1`
2. **One concept per test**: Test one thing at a time
3. **Use fixtures**: Reuse common setup via fixtures
4. **Test edge cases**: Empty inputs, invalid parameters, boundary conditions
5. **Check types and shapes**: Verify outputs have correct types and dimensions
6. **Use assertions liberally**: More assertions = better test coverage
7. **Add docstrings**: Explain what each test does

### Marking Tests

```python
@pytest.mark.slow
def test_expensive_computation():
    """This test takes a long time."""
    pass

@pytest.mark.integration
def test_full_workflow():
    """Tests multiple components together."""
    pass
```

## Common Patterns

### Testing Randomness

```python
def test_reproducibility(self, seed):
    """Test that same seed gives same results."""
    torch.manual_seed(seed)
    result1 = generate_random_data()
    
    torch.manual_seed(seed)
    result2 = generate_random_data()
    
    assert torch.allclose(result1, result2)
```

### Testing Shapes

```python
def test_output_shape(self):
    """Test output has correct shape."""
    output = model.forward(input)
    
    expected_shape = (batch_size, output_dim)
    assert output.shape == expected_shape
```

### Testing Errors

```python
def test_invalid_input_raises_error(self):
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError, match="must be positive"):
        model = Model(n_nodes=-1)
```

### Testing Numerical Values

```python
def test_reconstruction_quality(self):
    """Test reconstruction is close to original."""
    reconstructed = model.reconstruct(data)
    
    mse = ((data - reconstructed) ** 2).mean()
    assert mse < 0.01  # Threshold
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest --cov=src
```

## Troubleshooting

### Tests Fail Randomly

- Check for unset random seeds
- Look for race conditions (if using parallel tests)
- Verify fixtures are properly isolated

### Import Errors

- Make sure you're running from project root
- Check that `src/` is in PYTHONPATH
- Install package in editable mode: `pip install -e .`

### Slow Tests

- Use `pytest -m "not slow"` to skip slow tests during development
- Consider using pytest-xdist for parallel execution
- Profile tests with `pytest --durations=10`

### Coverage Issues

- Install pytest-cov: `pip install pytest-cov`
- Make sure all source files are imported
- Check `.coveragerc` configuration

## Continuous Improvement

### Adding Tests

When adding new features:
1. Write tests first (TDD)
2. Ensure >80% code coverage
3. Test both success and failure cases
4. Update this README if adding new test categories

### Maintaining Tests

- Run tests before committing
- Keep tests fast where possible
- Refactor tests along with code
- Remove obsolete tests

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [PyTorch Testing](https://pytorch.org/docs/stable/testing.html)