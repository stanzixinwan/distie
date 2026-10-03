#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "distie/block_pool.h"

namespace py = pybind11;

PYBIND11_MODULE(distie_core, m) {
  m.doc() = "DistIE C++ core (KV block metadata)";

  py::class_<distie::BlockPool>(m, "BlockPool")
      .def(py::init<std::size_t>(), py::arg("num_blocks"),
           "Track num_blocks KV page IDs; holds no tensor memory.")
      .def("allocate", &distie::BlockPool::Allocate, py::arg("count"),
           "Return count block IDs. Raises RuntimeError if the pool is short.")
      .def("free", &distie::BlockPool::Free, py::arg("block_ids"),
           "Return IDs to the pool. Raises ValueError on double-free.")
      .def_property_readonly("num_blocks", &distie::BlockPool::num_blocks)
      .def_property_readonly("num_free", &distie::BlockPool::num_free)
      .def_property_readonly("num_used", &distie::BlockPool::num_used)
      .def_property_readonly("utilization", &distie::BlockPool::utilization);
}
