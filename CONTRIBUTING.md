# Contributing to MCMC Bandits

We welcome contributions to improve this project! Here's how you can help:

## Reporting Issues

Found a bug or have a feature request? Please open an issue with the following information:
- A clear description of the issue/feature
- Steps to reproduce the issue (if applicable)
- Expected vs. actual behavior
- Any relevant error messages or screenshots

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/SarahLiaw/MCMC_cb.git
   cd MCMC_cb
   ```
3. Set up the development environment:
   ```bash
   conda env create -f environment.yml
   conda activate MCMC_cb
   pip install -e .
   ```
4. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use docstrings for all public functions and classes following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep lines under 88 characters (Black's default line length)
- Type hints are encouraged for better code clarity

## Testing

- Add tests for new features or bug fixes
- Run tests locally before submitting a pull request:
  ```bash
  pytest tests/
  ```
- Ensure all tests pass before pushing your changes

## Submitting Changes

1. Commit your changes with a clear and descriptive commit message
2. Push your branch to your fork
3. Open a pull request against the main branch
4. Ensure all CI checks pass
5. Address any code review feedback

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
