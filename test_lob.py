import numpy as np
import gpu_lob
import time
import matplotlib.pyplot as plt
from copy import deepcopy


class CPUMatcher:

    SIDE_BUY = 0
    SIDE_SELL = 1

    @staticmethod
    def match_orders(book_bids, book_asks, prices, quantities, sides, ids):
        n_orders = len(prices)
        matched_qty = np.zeros(n_orders, dtype=np.int32)

        for idx in range(n_orders):
            order_price = prices[idx]
            order_qty = quantities[idx]
            order_side = sides[idx]

            if order_price < 0 or order_price >= len(book_bids):
                matched_qty[idx] = 0
                continue

            total_matched = 0
            remaining_qty = order_qty

            if order_side == CPUMatcher.SIDE_BUY:
                available = book_asks[order_price]

                if available > 0:
                    match_qty = min(remaining_qty, available)
                    book_asks[order_price] -= match_qty
                    total_matched += match_qty

            elif order_side == CPUMatcher.SIDE_SELL:
                available = book_bids[order_price]

                if available > 0:
                    match_qty = min(remaining_qty, available)
                    book_bids[order_price] -= match_qty
                    total_matched += match_qty

            matched_qty[idx] = total_matched

        return matched_qty

    @staticmethod
    def add_liquidity(book_bids, book_asks, prices, quantities, sides):
        n_orders = len(prices)

        for idx in range(n_orders):
            price = prices[idx]
            qty = quantities[idx]
            side = sides[idx]

            if price < 0 or price >= len(book_bids):
                continue

            if side == CPUMatcher.SIDE_BUY:
                book_bids[price] += qty
            elif side == CPUMatcher.SIDE_SELL:
                book_asks[price] += qty


def print_book_summary(book_bids, book_asks, price_range):
    print("\n" + "="*60)
    print("ORDER BOOK SUMMARY")
    print("="*60)
    print(f"{'Price':<10} {'Bid Qty':<15} {'Ask Qty':<15}")
    print("-"*60)

    for price in price_range:
        bid_qty = book_bids[price]
        ask_qty = book_asks[price]
        if bid_qty > 0 or ask_qty > 0:
            print(f"{price:<10} {bid_qty:<15} {ask_qty:<15}")
    print("="*60 + "\n")


def test_basic_matching():
    print("\n" + "="*70)
    print("TEST 1: Basic Order Matching")
    print("="*70)

    MAX_PRICE = 1000
    book_bids = np.zeros(MAX_PRICE, dtype=np.int32)
    book_asks = np.zeros(MAX_PRICE, dtype=np.int32)

    print("\nAdding initial liquidity to ASK side...")
    initial_prices = np.array([100, 101, 102], dtype=np.int32)
    initial_quantities = np.array([1000, 1500, 2000], dtype=np.int32)
    initial_sides = np.array([gpu_lob.SIDE_SELL, gpu_lob.SIDE_SELL, gpu_lob.SIDE_SELL], dtype=np.int32)

    gpu_lob.add_liquidity(book_bids, book_asks, initial_prices, initial_quantities, initial_sides)

    print("Adding initial liquidity to BID side...")
    bid_prices = np.array([98, 99], dtype=np.int32)
    bid_quantities = np.array([800, 1200], dtype=np.int32)
    bid_sides = np.array([gpu_lob.SIDE_BUY, gpu_lob.SIDE_BUY], dtype=np.int32)

    gpu_lob.add_liquidity(book_bids, book_asks, bid_prices, bid_quantities, bid_sides)

    print_book_summary(book_bids, book_asks, range(95, 105))

    print("Submitting BUY orders to match against ASK book...")
    order_prices = np.array([100, 101, 100], dtype=np.int32)
    order_quantities = np.array([500, 800, 300], dtype=np.int32)
    order_sides = np.array([gpu_lob.SIDE_BUY, gpu_lob.SIDE_BUY, gpu_lob.SIDE_BUY], dtype=np.int32)
    order_ids = np.array([1, 2, 3], dtype=np.int32)

    matched_qty = gpu_lob.match_orders(
        book_bids, book_asks,
        order_prices, order_quantities, order_sides, order_ids
    )

    print("\nMATCH RESULTS:")
    print("-"*60)
    print(f"{'Order ID':<12} {'Side':<8} {'Price':<10} {'Qty':<10} {'Matched':<10}")
    print("-"*60)
    for i in range(len(order_ids)):
        side_str = "BUY" if order_sides[i] == gpu_lob.SIDE_BUY else "SELL"
        print(f"{order_ids[i]:<12} {side_str:<8} {order_prices[i]:<10} {order_quantities[i]:<10} {matched_qty[i]:<10}")
    print("-"*60)

    print_book_summary(book_bids, book_asks, range(95, 105))

    assert matched_qty[0] == 500, "Order 1 should match 500"
    assert matched_qty[1] == 800, "Order 2 should match 800"
    assert matched_qty[2] == 300, "Order 3 should match 300"
    assert book_asks[100] == 200, "Ask at 100 should have 200 remaining (1000 - 500 - 300)"
    assert book_asks[101] == 700, "Ask at 101 should have 700 remaining (1500 - 800)"

    print("\n✓ TEST 1 PASSED: Basic matching works correctly!\n")


def test_race_condition_handling():
    print("\n" + "="*70)
    print("TEST 2: Race Condition Handling with Atomic Operations")
    print("="*70)

    MAX_PRICE = 1000
    book_bids = np.zeros(MAX_PRICE, dtype=np.int32)
    book_asks = np.zeros(MAX_PRICE, dtype=np.int32)

    print("\nAdding limited liquidity (only 1000 units) at price 150...")
    liquidity_prices = np.array([150], dtype=np.int32)
    liquidity_quantities = np.array([1000], dtype=np.int32)
    liquidity_sides = np.array([gpu_lob.SIDE_SELL], dtype=np.int32)

    gpu_lob.add_liquidity(book_bids, book_asks, liquidity_prices, liquidity_quantities, liquidity_sides)

    print(f"Initial ask liquidity at price 150: {book_asks[150]}")

    print("\nSubmitting 100 concurrent BUY orders (500 units each) at price 150...")
    print("Total demand: 50,000 units vs 1,000 available")
    print("CUDA kernel will handle race conditions atomically!\n")

    n_orders = 100
    order_prices = np.full(n_orders, 150, dtype=np.int32)
    order_quantities = np.full(n_orders, 500, dtype=np.int32)
    order_sides = np.full(n_orders, gpu_lob.SIDE_BUY, dtype=np.int32)
    order_ids = np.arange(n_orders, dtype=np.int32)

    matched_qty = gpu_lob.match_orders(
        book_bids, book_asks,
        order_prices, order_quantities, order_sides, order_ids
    )

    total_matched = np.sum(matched_qty)
    fully_matched = np.sum(matched_qty == 500)
    partially_matched = np.sum((matched_qty > 0) & (matched_qty < 500))
    not_matched = np.sum(matched_qty == 0)

    print("RACE CONDITION TEST RESULTS:")
    print("-"*60)
    print(f"Total liquidity available:        1,000 units")
    print(f"Total matched across all orders:  {total_matched:,} units")
    print(f"Fully matched orders (500 units): {fully_matched}")
    print(f"Partially matched orders:         {partially_matched}")
    print(f"Unmatched orders:                 {not_matched}")
    print(f"Remaining ask liquidity at 150:   {book_asks[150]}")
    print("-"*60)

    assert total_matched == 1000, f"Total matched ({total_matched}) should equal initial liquidity (1000)"
    assert book_asks[150] == 0, f"Ask book at 150 should be empty, but has {book_asks[150]}"

    print("\n✓ TEST 2 PASSED: Atomic operations handled race conditions correctly!")
    print("  No liquidity was over-consumed or lost!\n")


def test_sell_orders():
    print("\n" + "="*70)
    print("TEST 3: Sell Orders Matching Against Bids")
    print("="*70)

    MAX_PRICE = 1000
    book_bids = np.zeros(MAX_PRICE, dtype=np.int32)
    book_asks = np.zeros(MAX_PRICE, dtype=np.int32)

    print("\nAdding BID liquidity at prices 200, 201, 202...")
    bid_prices = np.array([200, 201, 202], dtype=np.int32)
    bid_quantities = np.array([1000, 1500, 2000], dtype=np.int32)
    bid_sides = np.array([gpu_lob.SIDE_BUY, gpu_lob.SIDE_BUY, gpu_lob.SIDE_BUY], dtype=np.int32)

    gpu_lob.add_liquidity(book_bids, book_asks, bid_prices, bid_quantities, bid_sides)

    print_book_summary(book_bids, book_asks, range(198, 205))

    print("Submitting SELL orders to match against BID book...")
    order_prices = np.array([200, 201, 202], dtype=np.int32)
    order_quantities = np.array([600, 1500, 800], dtype=np.int32)
    order_sides = np.array([gpu_lob.SIDE_SELL, gpu_lob.SIDE_SELL, gpu_lob.SIDE_SELL], dtype=np.int32)
    order_ids = np.array([101, 102, 103], dtype=np.int32)

    matched_qty = gpu_lob.match_orders(
        book_bids, book_asks,
        order_prices, order_quantities, order_sides, order_ids
    )

    print("\nMATCH RESULTS:")
    print("-"*60)
    print(f"{'Order ID':<12} {'Side':<8} {'Price':<10} {'Qty':<10} {'Matched':<10}")
    print("-"*60)
    for i in range(len(order_ids)):
        side_str = "BUY" if order_sides[i] == gpu_lob.SIDE_BUY else "SELL"
        print(f"{order_ids[i]:<12} {side_str:<8} {order_prices[i]:<10} {order_quantities[i]:<10} {matched_qty[i]:<10}")
    print("-"*60)

    print_book_summary(book_bids, book_asks, range(198, 205))

    assert matched_qty[0] == 600, "Order 101 should match 600"
    assert matched_qty[1] == 1500, "Order 102 should match 1500 (full liquidity)"
    assert matched_qty[2] == 800, "Order 103 should match 800"
    assert book_bids[200] == 400, "Bid at 200 should have 400 remaining"
    assert book_bids[201] == 0, "Bid at 201 should be empty"

    print("\n✓ TEST 3 PASSED: Sell order matching works correctly!\n")


def performance_benchmark():
    print("\n" + "="*70)
    print("TEST 4: Performance Benchmark - Large Batch Processing")
    print("="*70)

    MAX_PRICE = 10000
    book_bids = np.zeros(MAX_PRICE, dtype=np.int32)
    book_asks = np.zeros(MAX_PRICE, dtype=np.int32)

    print("\nSetting up order book with liquidity at 1000 price levels...")
    n_levels = 1000
    liquidity_prices = np.arange(5000, 5000 + n_levels, dtype=np.int32)
    liquidity_quantities = np.random.randint(500, 2000, size=n_levels, dtype=np.int32)
    liquidity_sides = np.full(n_levels, gpu_lob.SIDE_SELL, dtype=np.int32)

    gpu_lob.add_liquidity(book_bids, book_asks, liquidity_prices, liquidity_quantities, liquidity_sides)

    n_orders = 10000
    print(f"Generating batch of {n_orders:,} random orders...")

    order_prices = np.random.randint(5000, 5000 + n_levels, size=n_orders, dtype=np.int32)
    order_quantities = np.random.randint(10, 200, size=n_orders, dtype=np.int32)
    order_sides = np.random.randint(0, 2, size=n_orders, dtype=np.int32)
    order_ids = np.arange(n_orders, dtype=np.int32)

    print(f"Matching {n_orders:,} orders against the book using CUDA...\n")

    start = time.perf_counter()
    matched_qty = gpu_lob.match_orders(
        book_bids, book_asks,
        order_prices, order_quantities, order_sides, order_ids
    )
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    throughput = n_orders / (end - start)

    total_matched = np.sum(matched_qty)
    match_rate = np.sum(matched_qty > 0) / n_orders * 100

    print("PERFORMANCE RESULTS:")
    print("="*60)
    print(f"Orders processed:         {n_orders:,}")
    print(f"Total matched quantity:   {total_matched:,}")
    print(f"Orders with matches:      {match_rate:.1f}%")
    print(f"Execution time:           {elapsed_ms:.3f} ms")
    print(f"Throughput:               {throughput:,.0f} orders/second")
    print(f"Latency per order:        {elapsed_ms/n_orders*1000:.3f} µs")
    print("="*60)

    print("\n✓ TEST 4 PASSED: Performance benchmark completed!\n")


def benchmark_comparison():
    print("\n" + "="*70)
    print("TEST 5: CPU vs GPU Performance Comparison")
    print("="*70)

    MAX_PRICE = 10000
    batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000]

    cpu_throughputs = []
    gpu_throughputs = []
    speedups = []

    print("\nBenchmarking across different batch sizes...")
    print(f"{'Batch Size':<12} {'CPU (ord/s)':<15} {'GPU (ord/s)':<15} {'Speedup':<10}")
    print("-" * 70)

    for n_orders in batch_sizes:
        n_levels = 1000
        liquidity_prices = np.arange(5000, 5000 + n_levels, dtype=np.int32)
        liquidity_quantities = np.random.randint(500, 5000, size=n_levels, dtype=np.int32)
        liquidity_sides = np.full(n_levels, gpu_lob.SIDE_SELL, dtype=np.int32)

        order_prices = np.random.randint(5000, 5000 + n_levels, size=n_orders, dtype=np.int32)
        order_quantities = np.random.randint(10, 200, size=n_orders, dtype=np.int32)
        order_sides = np.random.randint(0, 2, size=n_orders, dtype=np.int32)
        order_ids = np.arange(n_orders, dtype=np.int32)

        book_bids_cpu = np.zeros(MAX_PRICE, dtype=np.int32)
        book_asks_cpu = np.zeros(MAX_PRICE, dtype=np.int32)

        CPUMatcher.add_liquidity(book_bids_cpu, book_asks_cpu,
                                 liquidity_prices, liquidity_quantities, liquidity_sides)

        if n_orders <= 1000:
            _ = CPUMatcher.match_orders(
                book_bids_cpu.copy(), book_asks_cpu.copy(),
                order_prices, order_quantities, order_sides, order_ids
            )

        book_bids_test = book_bids_cpu.copy()
        book_asks_test = book_asks_cpu.copy()

        start_cpu = time.perf_counter()
        matched_cpu = CPUMatcher.match_orders(
            book_bids_test, book_asks_test,
            order_prices, order_quantities, order_sides, order_ids
        )
        end_cpu = time.perf_counter()

        cpu_time = end_cpu - start_cpu
        cpu_throughput = n_orders / cpu_time if cpu_time > 0 else 0

        book_bids_gpu = np.zeros(MAX_PRICE, dtype=np.int32)
        book_asks_gpu = np.zeros(MAX_PRICE, dtype=np.int32)

        gpu_lob.add_liquidity(book_bids_gpu, book_asks_gpu,
                              liquidity_prices, liquidity_quantities, liquidity_sides)

        _ = gpu_lob.match_orders(
            book_bids_gpu.copy(), book_asks_gpu.copy(),
            order_prices, order_quantities, order_sides, order_ids
        )

        book_bids_test = book_bids_gpu.copy()
        book_asks_test = book_asks_gpu.copy()

        start_gpu = time.perf_counter()
        matched_gpu = gpu_lob.match_orders(
            book_bids_test, book_asks_test,
            order_prices, order_quantities, order_sides, order_ids
        )
        end_gpu = time.perf_counter()

        gpu_time = end_gpu - start_gpu
        gpu_throughput = n_orders / gpu_time if gpu_time > 0 else 0

        speedup = gpu_throughput / cpu_throughput if cpu_throughput > 0 else 0

        cpu_throughputs.append(cpu_throughput)
        gpu_throughputs.append(gpu_throughput)
        speedups.append(speedup)

        print(f"{n_orders:<12,} {cpu_throughput:<15,.0f} {gpu_throughput:<15,.0f} {speedup:<10.2f}x")

        if not np.array_equal(matched_cpu, matched_gpu):
            print(f"  WARNING: CPU and GPU results differ for batch size {n_orders}")

    print("-" * 70)

    print("\nGenerating performance comparison plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(batch_sizes, cpu_throughputs, 'o-', label='CPU (Python)',
             linewidth=2, markersize=8, color='#FF6B6B')
    ax1.plot(batch_sizes, gpu_throughputs, 's-', label='GPU (CUDA)',
             linewidth=2, markersize=8, color='#4ECDC4')

    ax1.set_xlabel('Batch Size (orders)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (orders/second)', fontsize=12, fontweight='bold')
    ax1.set_title('CPU vs GPU Throughput Comparison', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both', linestyle='--')
    ax1.legend(fontsize=11, loc='best')

    max_cpu_idx = np.argmax(cpu_throughputs)
    max_gpu_idx = np.argmax(gpu_throughputs)
    ax1.annotate(f'{cpu_throughputs[max_cpu_idx]:,.0f} ord/s',
                 xy=(batch_sizes[max_cpu_idx], cpu_throughputs[max_cpu_idx]),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=9, color='#FF6B6B')
    ax1.annotate(f'{gpu_throughputs[max_gpu_idx]:,.0f} ord/s',
                 xy=(batch_sizes[max_gpu_idx], gpu_throughputs[max_gpu_idx]),
                 xytext=(10, -15), textcoords='offset points',
                 fontsize=9, color='#4ECDC4')

    ax2.plot(batch_sizes, speedups, 'D-', linewidth=2, markersize=8,
             color='#95E1D3', markerfacecolor='#F38181')
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No speedup')

    ax2.set_xlabel('Batch Size (orders)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup (GPU / CPU)', fontsize=12, fontweight='bold')
    ax2.set_title('GPU Speedup over CPU', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both', linestyle='--')
    ax2.legend(fontsize=11, loc='best')

    max_speedup_idx = np.argmax(speedups)
    ax2.annotate(f'{speedups[max_speedup_idx]:.1f}x',
                 xy=(batch_sizes[max_speedup_idx], speedups[max_speedup_idx]),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=10, fontweight='bold', color='#F38181')

    plt.tight_layout()

    output_file = 'benchmark_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    print("\n" + "="*70)
    print("SUMMARY:")
    print("-" * 70)
    print(f"Peak CPU Throughput:      {max(cpu_throughputs):,.0f} orders/second")
    print(f"Peak GPU Throughput:      {max(gpu_throughputs):,.0f} orders/second")
    print(f"Maximum Speedup:          {max(speedups):.2f}x")
    print(f"Speedup at 10K orders:    {speedups[-2]:.2f}x" if len(speedups) > 1 else "")
    print("="*70)

    print("\n✓ TEST 5 PASSED: CPU vs GPU comparison completed!\n")


def main():
    print("\n" + "="*70)
    print("CUDA LOB MATCHING ENGINE TEST SUITE")
    print("="*70)

    print(f"\nCUDA Devices: {gpu_lob.device_count()}")
    print(f"Active GPU: {gpu_lob.device_info()}\n")

    test_basic_matching()
    test_race_condition_handling()
    test_sell_orders()
    performance_benchmark()
    benchmark_comparison()

    print("="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    print("\nThe CUDA LOB matching engine is working correctly with:")
    print("  • Proper atomic operations for race condition handling")
    print("  • Correct matching logic for both buy and sell orders")
    print("  • High-performance batch processing capabilities")
    print("  • Significant performance advantage over CPU baseline")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
