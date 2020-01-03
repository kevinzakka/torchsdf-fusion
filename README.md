# torchsdf-fusion

My attempt at trying to outperform Andy Zeng's PyCuda [TSDF fusion](https://github.com/andyzeng/tsdf-fusion-python) implementation with PyTorch.

## Results

| Method          | FPS |
|-----------------|-----|
| PyCuda          | 50  |
| Vanilla PyTorch | 18  |
| C++ PyTorch     | 25  |
| JIT PyTorch     | 26  |