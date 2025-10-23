.PHONY: help test test-verbose test-quick posterior-analysis clean install build upload-test upload update-version

help:
	@echo "Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  make install           - Install dependencies from requirements.txt"
	@echo "  make install-dev       - Install package in editable mode with dev tools"
	@echo "  make test              - Run all tests"
	@echo "  make test-verbose      - Run tests with detailed output"
	@echo "  make test-quick        - Run tests excluding slow ones"
	@echo ""
	@echo "Running experiments:"
	@echo "  make posterior-analysis - Run posterior analysis with default settings"
	@echo "  make posterior-quick   - Run posterior analysis with LinTS and LMCTS only"
	@echo ""
	@echo "Package building:"
	@echo "  make build             - Build distribution packages"
	@echo "  make upload-test       - Upload to TestPyPI"
	@echo "  make upload            - Upload to PyPI (production)"
	@echo "  make install-local     - Install from local build"
	@echo "  make update-version    - Interactive version update script"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Remove generated files and cache"
	@echo "  make clean-all         - Remove generated files, cache, and builds"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e .[dev]

install-local:
	pip install .

test:
	pytest test_posterior_analysis.py -v

test-verbose:
	pytest test_posterior_analysis.py -vv --tb=long

test-quick:
	pytest test_posterior_analysis.py -v -m "not slow"

posterior-analysis:
	python posterior_analysis.py

posterior-quick:
	python posterior_analysis.py --algorithms LinTS LMCTS

build:
	@echo "Building distribution packages..."
	python -m build
	@echo "Build complete! Check dist/ directory"

upload-test:
	@echo "Uploading to TestPyPI..."
	python -m twine upload --repository testpypi dist/*

upload:
	@echo "Uploading to PyPI..."
	@echo "WARNING: This will publish to production PyPI!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		python -m twine upload dist/*; \
	fi

update-version:
	@./update_version.sh

clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf src/__pycache__
	rm -f *.pyc src/*.pyc
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf posterior_analysis_*

clean-all: clean
	rm -rf dist/ build/ *.egg-info
	rm -rf src/*.egg-info
