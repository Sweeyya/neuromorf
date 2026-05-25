<div align="center">

# neuromorf

**Compiles SNNs to run on neuromorphic hardware. All you need to know is PyTorch.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-48%20passing-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)]()

</div>

---

Neuromorphic chips like Intel Loihi 2 are fast, efficient, and powerful. But getting your trained model onto one means learning spike timing, membrane potentials, chip-specific SDKs, and a lot of other things that have nothing to do with your actual research. neuromorf handles all of that.

```python
import nir
import neuromorf

# load your trained SNN
nir_graph = nir.read("my_model.nir")

# compile it
compiled = neuromorf.compile(nir_graph, target="loihi2")

# run it
state = compiled.initialize_state()
output, state = compiled.run(input_data, state, num_timesteps=100)
```

---

## how it works

```
your model (NIR)
      |
   parse        convert NIR graph to neuromorf IR
      |
   validate     check structure + hardware compatibility
      |
   quantize     float32 weights to int8
      |
   codegen      generate runnable Lava / NumPy code
      |
neuromorphic hardware
```

---

## supported hardware

| chip | backend | status |
|------|---------|--------|
| Intel Loihi 2 | Lava | v1.0 |
| CPU (NumPy simulator) | NumPy | v1.0 |
| BrainChip Akida | coming soon | v0.3 |
| SpiNNaker | coming soon | v0.3 |

---

## supported neuron types

`IF` `LIF` `CubaLIF` `CubaLI` `LI` `I`

---

## install

```bash
pip install neuromorf
```

requires Python 3.10+

---

## status

v1.0 targets mid-July 2026.

| component | status |
|-----------|--------|
| IR (Neuron, Synapse, NeuromorphIR) | done |
| NIR parser | done |
| compiler passes | in progress |
| CPU backend | upcoming |
| Lava backend | upcoming |

---

## contributing

neuromorf is open source. if you want to add a backend for a new chip, open an issue first so we can discuss the IR mapping.

```bash
git clone https://github.com/Sweeyya/neuromorf
cd neuromorf
pip install -e ".[dev]"
pytest tests/
```

---

## built by

Sweeya, CS student at Georgia State University.
questions or ideas, open an issue.
