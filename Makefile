# Convenience targets.  Everything below can also be typed by hand (see README).
PY ?= python

.PHONY: venv env install install-cuda test test-gpu gates-cpu gates-gpu demo docker apptainer clean

venv:                       ## python -m venv .venv  (then: source .venv/bin/activate)
	$(PY) -m venv .venv && .venv/bin/pip install --upgrade pip setuptools wheel
	.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
	@echo "activate with:  source .venv/bin/activate && source env.sh"

env:                        ## conda env from environment.yml
	conda env create -f environment.yml

install:                    ## install the package into the current interpreter (CPU)
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

install-cuda:               ## add the CUDA 12 JAX wheels
	$(PY) -m pip install -r requirements-cuda.txt

test:                       ## unit tests, CPU only (a few minutes)
	JAX_PLATFORMS=cpu $(PY) -m pytest -q tests

test-gpu:                   ## the same tests on the GPU plus the GPU-only ones
	$(PY) -m pytest -q tests

gates-cpu:                  ## value gates (implicit == tiled, layer 0 == icet) on the CPU
	bash gates/run_gates_cpu.sh

gates-gpu:                  ## speed/memory gates and the central-entry smoke on the GPU
	bash gates/run_gates_gpu.sh

demo:                       ## 10-minute parallel-tempering demo of both models (needs a GPU)
	bash slurm/mc_demo.sh

docker:
	docker build -t neuralce .

apptainer:
	apptainer build --fakeroot neuralce.sif docker/neuralce.def

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache build *.egg-info
