# gpu_lob - CUDA/HIP-Accelerated Python Library

A high-performance CUDA/HIP-accelerated Python library for limit order book processing.
This project was made to work on Windows, do not try to build it on WSL or any Linux distribution.

## Prerequisites

- NVIDIA GPU with CUDA support
- AMD GPU with HIP support
- CUDA Toolkit (11.0+)
- ROCm (6.4)
- Visual Studio (2019 or 2022) C++ Build Tools
- Python 3.8+
- CMake 3.18+

## Build and Install

### 1. Create and Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
if you get an error about execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### 2. Upgrade pip and Install Build Tools

```powershell
pip install --upgrade pip
pip install scikit-build-core pybind11 numpy
```

### 3. Build for Your Platform

#### For NVIDIA GPUs (CUDA)
**Windows/Linux:**
```powershell
cp CMakeLists.txt.cuda_backup CMakeLists.txt
pip install -e .
```

#### For AMD GPUs (HIP) - Windows
**Windows:**
```powershell
.\build_windows.ps1
```

## Quick Start (Usage Example)

Once installed, you can use the library from Python:

```python
import gpu_lob
import numpy as np

MAX_PRICE = 1000
bids = np.zeros(MAX_PRICE, dtype=np.int32)
asks = np.zeros(MAX_PRICE, dtype=np.int32)

prices = np.array([100, 105], dtype=np.int32)
qtys = np.array([1000, 500], dtype=np.int32)
sides = np.array([gpu_lob.SIDE_BUY, gpu_lob.SIDE_SELL], dtype=np.int32)

gpu_lob.add_liquidity(bids, asks, prices, qtys, sides)

order_prices = np.array([105], dtype=np.int32)
order_qtys = np.array([200], dtype=np.int32)
order_sides = np.array([gpu_lob.SIDE_BUY], dtype=np.int32)
order_ids = np.array([1], dtype=np.int32)

matches = gpu_lob.match_orders(bids, asks, order_prices, order_qtys, order_sides, order_ids)

print(f"Matched: {matches[0]} units")
print(f"Remaining Ask at $105: {asks[105]}")
```

### 4. Install in Editable Mode
If you are modifying the C++ or CUDA code, use:
```powershell
pip install -e . -v --no-build-isolation
```

## Run Benchmarks

```powershell
python test_lob.py
```

Check the generated `benchmark_plot.png` for performance results.
