#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>

namespace py = pybind11;

extern "C" {
    void run_hello_kernel(int *output, int n);
    int get_device_count();
    void get_device_info(char *name, int maxLen);
}

extern "C" {
    void launch_matching_kernel(
        int* h_book_bids,
        int* h_book_asks,
        const int* h_prices,
        const int* h_quantities,
        const int* h_sides,
        const int* h_ids,
        int* h_matched_qty,
        int n_orders,
        int max_price
    );

    void add_liquidity(
        int* h_book_bids,
        int* h_book_asks,
        const int* h_prices,
        const int* h_quantities,
        const int* h_sides,
        int n_orders,
        int max_price
    );
}

py::array_t<int> run_cuda_hello(int n) {
    auto result = py::array_t<int>(n);
    py::buffer_info buf = result.request();
    int *ptr = static_cast<int*>(buf.ptr);

    run_hello_kernel(ptr, n);

    return result;
}

int cuda_device_count() {
    return get_device_count();
}

std::string cuda_device_info() {
    char name[256];
    get_device_info(name, sizeof(name));
    return std::string(name);
}

py::array_t<int> match_orders(
    py::array_t<int> book_bids,
    py::array_t<int> book_asks,
    py::array_t<int> prices,
    py::array_t<int> quantities,
    py::array_t<int> sides,
    py::array_t<int> ids
) {
    py::buffer_info bids_buf = book_bids.request();
    py::buffer_info asks_buf = book_asks.request();
    py::buffer_info prices_buf = prices.request();
    py::buffer_info quantities_buf = quantities.request();
    py::buffer_info sides_buf = sides.request();
    py::buffer_info ids_buf = ids.request();

    if (prices_buf.ndim != 1 || quantities_buf.ndim != 1 ||
        sides_buf.ndim != 1 || ids_buf.ndim != 1) {
        throw std::runtime_error("Order arrays must be 1-dimensional");
    }

    if (bids_buf.ndim != 1 || asks_buf.ndim != 1) {
        throw std::runtime_error("Order book arrays must be 1-dimensional");
    }

    int n_orders = prices_buf.shape[0];
    if (quantities_buf.shape[0] != n_orders ||
        sides_buf.shape[0] != n_orders ||
        ids_buf.shape[0] != n_orders) {
        throw std::runtime_error("All order arrays must have the same length");
    }

    int max_price = bids_buf.shape[0];
    if (asks_buf.shape[0] != max_price) {
        throw std::runtime_error("Bid and ask book arrays must have the same length");
    }

    int* bids_ptr = static_cast<int*>(bids_buf.ptr);
    int* asks_ptr = static_cast<int*>(asks_buf.ptr);
    int* prices_ptr = static_cast<int*>(prices_buf.ptr);
    int* quantities_ptr = static_cast<int*>(quantities_buf.ptr);
    int* sides_ptr = static_cast<int*>(sides_buf.ptr);
    int* ids_ptr = static_cast<int*>(ids_buf.ptr);

    auto matched_qty = py::array_t<int>(n_orders);
    py::buffer_info matched_buf = matched_qty.request();
    int* matched_ptr = static_cast<int*>(matched_buf.ptr);

    launch_matching_kernel(
        bids_ptr,
        asks_ptr,
        prices_ptr,
        quantities_ptr,
        sides_ptr,
        ids_ptr,
        matched_ptr,
        n_orders,
        max_price
    );

    return matched_qty;
}

void add_liquidity_to_book(
    py::array_t<int> book_bids,
    py::array_t<int> book_asks,
    py::array_t<int> prices,
    py::array_t<int> quantities,
    py::array_t<int> sides
) {
    py::buffer_info bids_buf = book_bids.request();
    py::buffer_info asks_buf = book_asks.request();
    py::buffer_info prices_buf = prices.request();
    py::buffer_info quantities_buf = quantities.request();
    py::buffer_info sides_buf = sides.request();

    if (prices_buf.ndim != 1 || quantities_buf.ndim != 1 || sides_buf.ndim != 1) {
        throw std::runtime_error("Order arrays must be 1-dimensional");
    }

    if (bids_buf.ndim != 1 || asks_buf.ndim != 1) {
        throw std::runtime_error("Order book arrays must be 1-dimensional");
    }

    int n_orders = prices_buf.shape[0];
    if (quantities_buf.shape[0] != n_orders || sides_buf.shape[0] != n_orders) {
        throw std::runtime_error("All order arrays must have the same length");
    }

    int max_price = bids_buf.shape[0];
    if (asks_buf.shape[0] != max_price) {
        throw std::runtime_error("Bid and ask book arrays must have the same length");
    }

    int* bids_ptr = static_cast<int*>(bids_buf.ptr);
    int* asks_ptr = static_cast<int*>(asks_buf.ptr);
    int* prices_ptr = static_cast<int*>(prices_buf.ptr);
    int* quantities_ptr = static_cast<int*>(quantities_buf.ptr);
    int* sides_ptr = static_cast<int*>(sides_buf.ptr);

    add_liquidity(
        bids_ptr,
        asks_ptr,
        prices_ptr,
        quantities_ptr,
        sides_ptr,
        n_orders,
        max_price
    );
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "CUDA-accelerated limit order book processing library";

    m.def("run_cuda_hello", &run_cuda_hello,
          py::arg("n"),
          "Run a simple CUDA kernel that computes squares of indices.\n\n"
          "Args:\n"
          "    n: Number of elements to process\n\n"
          "Returns:\n"
          "    numpy.ndarray: Array of computed values");

    m.def("device_count", &cuda_device_count,
          "Get the number of available CUDA devices.\n\n"
          "Returns:\n"
          "    int: Number of CUDA devices");

    m.def("device_info", &cuda_device_info,
          "Get information about the current CUDA device.\n\n"
          "Returns:\n"
          "    str: Device name and properties");

    m.def("match_orders", &match_orders,
          py::arg("book_bids"),
          py::arg("book_asks"),
          py::arg("prices"),
          py::arg("quantities"),
          py::arg("sides"),
          py::arg("ids"),
          "Match a batch of orders against the limit order book using CUDA.\n\n"
          "Args:\n"
          "    book_bids: numpy array of bid quantities (modified in-place)\n"
          "    book_asks: numpy array of ask quantities (modified in-place)\n"
          "    prices: numpy array of order prices\n"
          "    quantities: numpy array of order quantities\n"
          "    sides: numpy array of order sides (0=BUY, 1=SELL)\n"
          "    ids: numpy array of order IDs\n\n"
          "Returns:\n"
          "    numpy.ndarray: Matched quantities for each order\n\n"
          "Note:\n"
          "    The book_bids and book_asks arrays are modified in-place to reflect\n"
          "    the liquidity consumed by the matched orders. Race conditions are\n"
          "    handled atomically using CUDA atomic operations.");

    m.def("add_liquidity", &add_liquidity_to_book,
          py::arg("book_bids"),
          py::arg("book_asks"),
          py::arg("prices"),
          py::arg("quantities"),
          py::arg("sides"),
          "Add liquidity to the order book using CUDA.\n\n"
          "Args:\n"
          "    book_bids: numpy array of bid quantities (modified in-place)\n"
          "    book_asks: numpy array of ask quantities (modified in-place)\n"
          "    prices: numpy array of order prices\n"
          "    quantities: numpy array of order quantities to add\n"
          "    sides: numpy array of order sides (0=BUY, 1=SELL)\n\n"
          "Note:\n"
          "    This function atomically adds the specified quantities to the\n"
          "    order book at the given price levels.");

    m.attr("SIDE_BUY") = 0;
    m.attr("SIDE_SELL") = 1;

    m.attr("__version__") = "0.1.0";
}
