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

| Method          | FPS |
|-----------------|-----|
| PyCuda          | 50  |
| Vanilla PyTorch | 18  |
| C++ PyTorch     | 25  |
| JIT PyTorch     | 26  |