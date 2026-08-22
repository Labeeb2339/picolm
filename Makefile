.PHONY: venv install train generate test demo

ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif


venv:               ## create the local virtual environment
	python -m venv .venv

install: venv       ## install CUDA 12.8 torch + package into .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cu128
	$(PY) -m pip install -e ".[dev,demo]"

train:              ## train the demo model on tiny Shakespeare
	$(PY) -m picolm train --text data/input.txt --out-dir out --max-iters 5000

generate:           ## sample from the trained checkpoint
	$(PY) -m picolm generate --ckpt out/ckpt.pt --prompt "To be, or not to be"

test:               ## run the test suite
	$(PY) -m pytest

demo:               ## launch the Streamlit demo
	$(PY) -m picolm demo
