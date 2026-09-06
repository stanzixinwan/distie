#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "distie/block_pool.h"

namespace py = pybind11;

PYBIND11_MODULE(distie_core, m) {
  m.doc() = "DistIE C++ performance core (block memory pool)";

  py::class_<distie::BlockPool>(m, "BlockPool")
      .def(py::init<std::size_t, std::size_t>(), py::arg("num_blocks"),
           py::arg("block_size_bytes"),
           "Pre-allocate num_blocks slabs of block_size_bytes each.")
      .def("allocate", &distie::BlockPool::Allocate, py::arg("count"),
           "Return count block IDs. Raises RuntimeError if the pool is short.")
      .def("free", &distie::BlockPool::Free, py::arg("block_ids"),
           "Return IDs to the pool. Raises ValueError on double-free.")
      .def(
          "block_view",
          [](distie::BlockPool& pool, distie::BlockId id) {
            return py::memoryview::from_memory(
                pool.MutableBlock(id),
                static_cast<py::ssize_t>(pool.block_size_bytes()),
                /*readonly=*/false);
          },
          py::arg("block_id"), py::keep_alive<0, 1>(),
          "Writable memoryview of one block. Do not use after free().")
      .def_property_readonly("num_blocks", &distie::BlockPool::num_blocks)
      .def_property_readonly("block_size_bytes", &distie::BlockPool::block_size_bytes)
      .def_property_readonly("num_free", &distie::BlockPool::num_free)
      .def_property_readonly("num_used", &distie::BlockPool::num_used)
      .def_property_readonly("utilization", &distie::BlockPool::utilization);
}
