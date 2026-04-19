# Security Policy

## Reporting a Vulnerability

If you believe you have found a security issue in safetensors, please do **not** open a public GitHub issue.

Instead, email [security@huggingface.co](mailto:security@huggingface.co) with a description of the issue, steps to reproduce, and any relevant details. Someone from the Hugging Face security team will review your report and recommend next steps.

You may also disclose your report through [Huntr](https://huntr.com), a vulnerability disclosure program for open-source projects.

## Hugging Face Hub and remote artefacts

Safetensors is open-source software that defines a simple format for storing tensors safely, and provides fast
(zero-copy) Rust and Python libraries to read and write it. It was designed specifically to prevent the arbitrary
code execution risks associated with formats like [pickle](https://docs.python.org/3/library/pickle.html).

While safetensors can be used fully offline, it is commonly paired with the Hugging Face Hub to download model
weights uploaded by others. When consuming artefacts from any platform, you expose yourself to risks. The
recommendations below help keep your runtime and local environment safe.

### Remote artefacts

Models uploaded on the Hugging Face Hub come in different formats. We heavily recommend uploading and downloading
models in the [`safetensors`](https://github.com/huggingface/safetensors) format, which cannot execute arbitrary code when loaded.

When loading a model through a downstream library that supports multiple formats (e.g. `transformers`, `diffusers`),
prefer the option that forces the use of safetensors (such as `use_safetensors=True` in `transformers`) so that
loading fails loudly rather than silently falling back to an unsafe format.

We also recommend pinning a specific revision of the repository you download from, to protect yourself from
upstream changes to the weights.
