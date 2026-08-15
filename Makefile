.PHONY: install train generate test demo

PY := .venv/Scripts/python
PIP := .venv/Scripts/pip

install:            ## create venv + install CUDA torch + package
	$(PY) -m pip install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu128
	$(PIP) install -e ".[dev,demo]"

train:              ## train the demo model on tiny Shakespeare
	$(PY) -m picolm train --text data/input.txt --out-dir out --max-iters 5000

generate:           ## sample from the trained checkpoint
	$(PY) -m picolm generate --ckpt out/ckpt.pt --prompt "To be, or not to be"

test:               ## run the test suite
	$(PY) -m pytest

demo:               ## launch the Streamlit demo
	$(PY) -m picolm demo
