## Installation

```
pip install safetensors
```


## Usage

### Numpy

```python
from safetensors.numpy import save_file, load_file
import numpy as np

tensors = {
   "a": np.zeros((2, 2)),
   "b": np.zeros((2, 3), dtype=np.uint8)
}

save_file(tensors, "./model.safetensors")


# Now loading
loaded = load_file("./model.safetensors")
```

### Torch

```python
from safetensors.torch import save_file, load_file
import torch

tensors = {
   "a": torch.zeros((2, 2)),
   "b": torch.zeros((2, 3), dtype=torch.uint8)
}

save_file(tensors, "./model.safetensors")


# Now loading
loaded = load_file("./model.safetensors")
```

When loading large models, some unified memory systems (such as Strix Halo) behave poorly with PyTorch's file-backed storage.
You may set `SAFETENSORS_DISABLE_TORCH_FROM_FILE=1` before loading.

### Developing

```
# inside ./safetensors/bindings/python
pip install .[dev]
```
Should be enough to install this library locally.

### Testing

```
# inside ./safetensors/bindings/python
pip install .[dev]
pytest -sv tests/
```
