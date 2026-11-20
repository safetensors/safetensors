# Convert weights to safetensors

PyTorch model weights are commonly saved and stored as `.bin` files with Python's [`pickle`](https://docs.python.org/3/library/pickle.html) utility. To save and store your model weights in the more secure `safetensor` format, we recommend converting your weights to `.safetensors`.

The easiest way to convert your model weights is to use the [Convert Space](https://huggingface.co/spaces/safetensors/convert), given your model weights are already stored on the Hub. The Convert Space downloads the pickled weights, converts them, and opens a Pull Request to upload the newly converted `.safetensors` file to your repository.

<Tip warning={true}>

For larger models, the Space may be a bit slower because its resources are tied up in converting other models. You can also try running the [convert.py](https://github.com/safetensors/safetensors/blob/main/bindings/python/convert.py) script (this is what the Space is running) locally to convert your weights.

Feel free to ping [@Narsil](https://huggingface.co/Narsil) for any issues with the Space.

</Tip>
