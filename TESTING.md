# Testing Guide

This document describes how to test the posterior analysis functionality.

## Quick Start

### Run All Tests

```bash
# Using pytest (recommended)
pytest test_posterior_analysis.py -v

# Using unittest
python test_posterior_analysis.py
```

### Run Specific Test Classes

```bash
# Test only data generation
pytest test_posterior_analysis.py::TestDataGeneration -v

# Test only posterior computation
pytest test_posterior_analysis.py::TestPosteriorComputation -v

# Test only feature maps
pytest test_posterior_analysis.py::TestFeatureMaps -v
```

### Run Specific Tests

```bash
# Test correlated context generation
pytest test_posterior_analysis.py::TestDataGeneration::test_correlated_contexts -v

# Test posterior with no data
pytest test_posterior_analysis.py::TestPosteriorComputation::test_prior_with_no_data -v
```

## Test Coverage

The test suite covers:

### 1. Data Generation (`TestDataGeneration`)
- **test_uncorrelated_contexts**: Validates i.i.d. N(0,I) context generation
- **test_correlated_contexts**: Validates correlated context generation with increased variance
- **test_reproducibility**: Ensures same seed produces identical data

### 2. Posterior Computation (`TestPosteriorComputation`)
- **test_prior_with_no_data**: Posterior equals prior with zero observations
- **test_posterior_with_single_observation**: Correct update with one data point
- **test_posterior_reduces_uncertainty**: More data decreases posterior variance

### 3. Feature Maps (`TestFeatureMaps`)
- **test_phi_linear_shape**: Block diagonal structure has correct dimensions
- **test_phi_a_linear_extracts_correct_block**: Single arm extraction works correctly

### 4. Posterior Sampling (`TestPosteriorSampling`)
- **test_sample_from_posterior_shape**: Samples have correct dimensions
- **test_sample_statistics**: Sample mean and covariance match distribution parameters

### 5. Wasserstein Distance (`TestWassersteinDistance`)
- **test_identical_distributions**: Zero distance for identical distributions
- **test_empty_samples**: Proper NaN handling for empty sets
- **test_mean_shift**: Distance increases with mean shift

### 6. Utilities (`TestSeedSetting`)
- **test_set_seed_reproducibility**: Random seeds ensure reproducibility

### 7. Integration (`TestIntegration`)
- **test_full_pipeline**: Complete workflow from data generation to sampling

## Expected Output

When all tests pass, you should see:

```
test_posterior_analysis.py::TestDataGeneration::test_uncorrelated_contexts PASSED
test_posterior_analysis.py::TestDataGeneration::test_correlated_contexts PASSED
test_posterior_analysis.py::TestDataGeneration::test_reproducibility PASSED
test_posterior_analysis.py::TestPosteriorComputation::test_prior_with_no_data PASSED
test_posterior_analysis.py::TestPosteriorComputation::test_posterior_with_single_observation PASSED
test_posterior_analysis.py::TestPosteriorComputation::test_posterior_reduces_uncertainty PASSED
test_posterior_analysis.py::TestFeatureMaps::test_phi_linear_shape PASSED
test_posterior_analysis.py::TestFeatureMaps::test_phi_a_linear_extracts_correct_block PASSED
test_posterior_analysis.py::TestPosteriorSampling::test_sample_from_posterior_shape PASSED
test_posterior_analysis.py::TestPosteriorSampling::test_sample_statistics PASSED
test_posterior_analysis.py::TestWassersteinDistance::test_identical_distributions PASSED
test_posterior_analysis.py::TestWassersteinDistance::test_empty_samples PASSED
test_posterior_analysis.py::TestWassersteinDistance::test_mean_shift PASSED
test_posterior_analysis.py::TestSeedSetting::test_set_seed_reproducibility PASSED
test_posterior_analysis.py::TestIntegration::test_full_pipeline PASSED

========================= 15 passed in X.XXs =========================
```

## Debugging Failed Tests

If tests fail:

1. **Read the error message carefully** - pytest provides detailed tracebacks
2. **Run the specific failing test** with more verbosity:
   ```bash
   pytest test_posterior_analysis.py::TestClass::test_name -vv
   ```
3. **Check numerical tolerances** - some tests use `atol` parameters that may need adjustment
4. **Verify dependencies** - ensure scipy, torch, numpy are properly installed

## Adding New Tests

To add a new test:

1. Create a new test class or add to existing class:
   ```python
   class TestNewFeature(unittest.TestCase):
       def test_my_feature(self):
           # Arrange
           input_data = ...
           
           # Act
           result = my_function(input_data)
           
           # Assert
           self.assertEqual(result, expected_value)
   ```

2. Follow naming conventions:
   - Test files: `test_*.py`
   - Test classes: `Test*`
   - Test methods: `test_*`

3. Use descriptive names explaining what is being tested

4. Include docstrings explaining the test's purpose

## Continuous Integration

To integrate with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest test_posterior_analysis.py -v --tb=short
```

## Performance Considerations

Some tests involve:
- Random sampling (5000+ samples for statistics tests)
- Matrix operations (Cholesky decomposition, matrix inversion)
- Multiple test repetitions

**Typical runtime**: 5-15 seconds for full test suite

To skip slow tests during development:
```bash
pytest test_posterior_analysis.py -v -m "not slow"
```

## Test Dependencies

Required packages:
- `torch` - Tensor operations and distributions
- `numpy` - Numerical computations
- `scipy` - Wasserstein distance (scipy.linalg)
- `pytest` - Test framework (optional, can use unittest)

## Troubleshooting

### Common Issues

**Import errors:**
```
ModuleNotFoundError: No module named 'posterior_analysis'
```
**Solution**: Run tests from repository root directory

**Numerical precision failures:**
```
AssertionError: Tensor not close enough
```
**Solution**: Tests use tolerances (`atol`, `rtol`). Adjust if needed for your platform.

**Random failures:**
```
Test passes sometimes, fails other times
```
**Solution**: Ensure all tests use `set_seed()` or `torch.manual_seed()` for reproducibility.

## Contact

For test-related issues, please open an issue on the repository with:
- Full error traceback
- Python version and platform
- Output of `pip list` showing package versions
