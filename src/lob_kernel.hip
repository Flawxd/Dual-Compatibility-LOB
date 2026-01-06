#include "gpu_compat.h"
#include <stdio.h>

// Order side enumeration
#define SIDE_BUY  0
#define SIDE_SELL 1

// Maximum price level (adjust based on your use case)
#define MAX_PRICE 10000

/**
 * GPU kernel for high-frequency limit order book matching
 *
 * @param book_bids      Bid side order book (quantity at each price level)
 * @param book_asks      Ask side order book (quantity at each price level)
 * @param prices         Input order prices (SoA)
 * @param quantities     Input order quantities (SoA)
 * @param sides          Input order sides: 0=BUY, 1=SELL (SoA)
 * @param ids            Input order IDs (SoA)
 * @param matched_qty    Output: matched quantities for each order
 * @param n_orders       Number of orders in the batch
 * @param max_price      Maximum valid price level
 */
__global__ void match_orders_kernel(
    int* book_bids,
    int* book_asks,
    const int* prices,
    const int* quantities,
    const int* sides,
    const int* ids,
    int* matched_qty,
    int n_orders,
    int max_price
) {
    // Each thread processes one order
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_orders) return;

    // Load order details for this thread
    int order_price = prices[idx];
    int order_qty = quantities[idx];
    int order_side = sides[idx];
    int order_id = ids[idx];

    // Validate price level
    if (order_price < 0 || order_price >= max_price) {
        matched_qty[idx] = 0;
        return;
    }

    int total_matched = 0;
    int remaining_qty = order_qty;

    if (order_side == SIDE_BUY) {
        // BUY order: match against ASK book at this price level
        // Try to take liquidity from the ask side

        while (remaining_qty > 0) {
            // Read current book quantity (volatile to prevent caching issues)
            int available = book_asks[order_price];

            if (available <= 0) {
                // No liquidity available at this price
                break;
            }

            // Determine how much we can match
            int match_qty = min(remaining_qty, available);

            // Atomically try to decrement the book
            int old_qty = atomicSub(&book_asks[order_price], match_qty);

            // Check for race condition: did we overdraw?
            if (old_qty < match_qty) {
                // Race condition detected: book quantity went negative
                // We need to reverse the excess withdrawal
                int overdraw = match_qty - old_qty;

                if (old_qty <= 0) {
                    // Completely failed - restore everything
                    atomicAdd(&book_asks[order_price], match_qty);
                    break;
                } else {
                    // Partial success - restore overdraw amount
                    atomicAdd(&book_asks[order_price], overdraw);
                    match_qty = old_qty;  // We only got what was available
                }
            }

            // Update our matched quantity
            total_matched += match_qty;
            remaining_qty -= match_qty;

            // If we matched something but still have remaining quantity,
            // the book is now empty at this price, so we're done
            if (remaining_qty > 0) {
                break;
            }
        }

    } else if (order_side == SIDE_SELL) {
        // SELL order: match against BID book at this price level
        // Try to take liquidity from the bid side

        while (remaining_qty > 0) {
            // Read current book quantity
            int available = book_bids[order_price];

            if (available <= 0) {
                // No liquidity available at this price
                break;
            }

            // Determine how much we can match
            int match_qty = min(remaining_qty, available);

            // Atomically try to decrement the book
            int old_qty = atomicSub(&book_bids[order_price], match_qty);

            // Check for race condition: did we overdraw?
            if (old_qty < match_qty) {
                // Race condition detected: book quantity went negative
                int overdraw = match_qty - old_qty;

                if (old_qty <= 0) {
                    // Completely failed - restore everything
                    atomicAdd(&book_bids[order_price], match_qty);
                    break;
                } else {
                    // Partial success - restore overdraw amount
                    atomicAdd(&book_bids[order_price], overdraw);
                    match_qty = old_qty;
                }
            }

            // Update our matched quantity
            total_matched += match_qty;
            remaining_qty -= match_qty;

            // If we matched something but still have remaining quantity,
            // the book is now empty at this price, so we're done
            if (remaining_qty > 0) {
                break;
            }
        }
    }

    // Write final matched quantity for this order
    matched_qty[idx] = total_matched;
}

/**
 * Host function to launch the matching kernel
 */
extern "C" void launch_matching_kernel(
    int* h_book_bids,
    int* h_book_asks,
    const int* h_prices,
    const int* h_quantities,
    const int* h_sides,
    const int* h_ids,
    int* h_matched_qty,
    int n_orders,
    int max_price
) {
    // Device pointers
    int *d_book_bids, *d_book_asks;
    int *d_prices, *d_quantities, *d_sides, *d_ids, *d_matched_qty;

    size_t book_size = max_price * sizeof(int);
    size_t orders_size = n_orders * sizeof(int);

    // Allocate device memory for order books
    checkGpuError(gpuMalloc(&d_book_bids, book_size));
    checkGpuError(gpuMalloc(&d_book_asks, book_size));

    // Allocate device memory for orders (SoA)
    checkGpuError(gpuMalloc(&d_prices, orders_size));
    checkGpuError(gpuMalloc(&d_quantities, orders_size));
    checkGpuError(gpuMalloc(&d_sides, orders_size));
    checkGpuError(gpuMalloc(&d_ids, orders_size));
    checkGpuError(gpuMalloc(&d_matched_qty, orders_size));

    // Copy input data to device
    checkGpuError(gpuMemcpy(d_book_bids, h_book_bids, book_size, gpuMemcpyHostToDevice));
    checkGpuError(gpuMemcpy(d_book_asks, h_book_asks, book_size, gpuMemcpyHostToDevice));
    checkGpuError(gpuMemcpy(d_prices, h_prices, orders_size, gpuMemcpyHostToDevice));
    checkGpuError(gpuMemcpy(d_quantities, h_quantities, orders_size, gpuMemcpyHostToDevice));
    checkGpuError(gpuMemcpy(d_sides, h_sides, orders_size, gpuMemcpyHostToDevice));
    checkGpuError(gpuMemcpy(d_ids, h_ids, orders_size, gpuMemcpyHostToDevice));

    // Initialize output array
    checkGpuError(gpuMemset(d_matched_qty, 0, orders_size));

    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_orders + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    match_orders_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_book_bids,
        d_book_asks,
        d_prices,
        d_quantities,
        d_sides,
        d_ids,
        d_matched_qty,
        n_orders,
        max_price
    );

    // Check for kernel launch errors
    checkGpuError(gpuGetLastError());

    // Wait for kernel to complete
    checkGpuError(gpuDeviceSynchronize());

    // Copy results back to host
    checkGpuError(gpuMemcpy(h_matched_qty, d_matched_qty, orders_size, gpuMemcpyDeviceToHost));
    checkGpuError(gpuMemcpy(h_book_bids, d_book_bids, book_size, gpuMemcpyDeviceToHost));
    checkGpuError(gpuMemcpy(h_book_asks, d_book_asks, book_size, gpuMemcpyDeviceToHost));

    // Free device memory
    checkGpuError(gpuFree(d_book_bids));
    checkGpuError(gpuFree(d_book_asks));
    checkGpuError(gpuFree(d_prices));
    checkGpuError(gpuFree(d_quantities));
    checkGpuError(gpuFree(d_sides));
    checkGpuError(gpuFree(d_ids));
    checkGpuError(gpuFree(d_matched_qty));
}

/**
 * Kernel to add liquidity to the order book (for testing/initialization)
 */
__global__ void add_liquidity_kernel(
    int* book_bids,
    int* book_asks,
    const int* prices,
    const int* quantities,
    const int* sides,
    int n_orders,
    int max_price
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_orders) return;

    int price = prices[idx];
    int qty = quantities[idx];
    int side = sides[idx];

    if (price < 0 || price >= max_price) return;

    if (side == SIDE_BUY) {
        atomicAdd(&book_bids[price], qty);
    } else if (side == SIDE_SELL) {
        atomicAdd(&book_asks[price], qty);
    }
}

/**
 * Host function to add liquidity to the order book
 */
extern "C" void add_liquidity(
    int* h_book_bids,
    int* h_book_asks,
    const int* h_prices,
    const int* h_quantities,
    const int* h_sides,
    int n_orders,
    int max_price
) {
    int *d_book_bids, *d_book_asks;
    int *d_prices, *d_quantities, *d_sides;

    size_t book_size = max_price * sizeof(int);
    size_t orders_size = n_orders * sizeof(int);

    checkGpuError(gpuMalloc(&d_book_bids, book_size));
    checkGpuError(gpuMalloc(&d_book_asks, book_size));
    checkGpuError(gpuMalloc(&d_prices, orders_size));
    checkGpuError(gpuMalloc(&d_quantities, orders_size));
    checkGpuError(gpuMalloc(&d_sides, orders_size));

    checkGpuError(gpuMemcpy(d_book_bids, h_book_bids, book_size, gpuMemcpyHostToDevice));
    checkGpuError(gpuMemcpy(d_book_asks, h_book_asks, book_size, gpuMemcpyHostToDevice));
    checkGpuError(gpuMemcpy(d_prices, h_prices, orders_size, gpuMemcpyHostToDevice));
    checkGpuError(gpuMemcpy(d_quantities, h_quantities, orders_size, gpuMemcpyHostToDevice));
    checkGpuError(gpuMemcpy(d_sides, h_sides, orders_size, gpuMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n_orders + threadsPerBlock - 1) / threadsPerBlock;

    add_liquidity_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_book_bids, d_book_asks, d_prices, d_quantities, d_sides, n_orders, max_price
    );

    checkGpuError(gpuGetLastError());
    checkGpuError(gpuDeviceSynchronize());

    checkGpuError(gpuMemcpy(h_book_bids, d_book_bids, book_size, gpuMemcpyDeviceToHost));
    checkGpuError(gpuMemcpy(h_book_asks, d_book_asks, book_size, gpuMemcpyDeviceToHost));

    checkGpuError(gpuFree(d_book_bids));
    checkGpuError(gpuFree(d_book_asks));
    checkGpuError(gpuFree(d_prices));
    checkGpuError(gpuFree(d_quantities));
    checkGpuError(gpuFree(d_sides));
}
