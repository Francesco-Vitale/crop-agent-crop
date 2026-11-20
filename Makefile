PY ?= python
.PHONY: install test reference partA partB plots all docker
install:
	pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt
test:
	$(PY) -m pytest -q tests
partA:
	$(PY) run_partA.py 0 1 2 3 4
partB:
	$(PY) run_partB.py
	$(PY) run_partB_sensitivity.py
reference:
	$(PY) run_ref.py
plots:
	$(PY) make_plots.py
all: test reference partA partB plots
docker:
	docker build -t crop-agent . && docker run --rm -v $$(pwd)/results:/app/results crop-agent
