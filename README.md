# torchsdf-fusion

My attempt at trying to outperform Andy Zeng's PyCuda [TSDF fusion](https://github.com/andyzeng/tsdf-fusion-python) implementation with PyTorch.

To build the PyTorch C++ integration function on macOS use:

```
CXX=clang python setup.py install
```

To run the benchmark, use:

```
python benchmark.py py/cpp/jit
```

## Results

| Method          | FPS | Avg Integration Time |
|-----------------|-----|----------------------|
| PyCuda          | 64.45|      0.006|
| Vanilla PyTorch | 24.77  |    0.031|
| C++ PyTorch     | 24.71  |    0.031|
| JIT PyTorch     | 26.01  |    0.029|