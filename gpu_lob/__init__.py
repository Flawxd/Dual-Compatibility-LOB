from gpu_lob._core import (
    run_cuda_hello,
    device_count,
    device_info,
    match_orders,
    add_liquidity,
    SIDE_BUY,
    SIDE_SELL,
    __version__
)

__all__ = [
    'run_cuda_hello',
    'device_count',
    'device_info',
    'match_orders',
    'add_liquidity',
    'SIDE_BUY',
    'SIDE_SELL',
    '__version__'
]


def test_installation():
    try:
        num_devices = device_count()
        print(f"CUDA devices found: {num_devices}")

        if num_devices > 0:
            info = device_info()
            print(f"Active device: {info}")

            result = run_cuda_hello(10)
            print(f"Test kernel output: {result}")
            print("Installation test PASSED!")
            return True
        else:
            print("WARNING: No CUDA devices found. Check your GPU and drivers.")
            return False
    except Exception as e:
        print(f"Installation test FAILED: {e}")
        return False
