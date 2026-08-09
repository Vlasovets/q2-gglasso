.PHONY: all lint dev install clean distclean test

all: install

lint:
	q2lint
	ruff check .

dev:
	pip install -e ".[dev]"

install:
	pip install -e .

# Tests use paths relative to the repo root, so always run from here.
# `install` stays a prerequisite: TestPluginBase resolves the plugin through the
# qiime2.plugins entry point, which only exists once the package is installed.
test: install
	python -m pytest q2_gglasso/tests -v --durations=20

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -exec rm -f {} +
	find . -name "*.pyo" -exec rm -f {} +
	find . -name "*.pyd" -exec rm -f {} +
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".coverage" -exec rm -f {} +

distclean: clean
