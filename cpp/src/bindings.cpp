#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gpt.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gpt_cpp, m)
{
    py::class_<GPT2Inference>(m, "GPT2Inference")
        .def(py::init<const std::string&>())
        .def("forward", &GPT2Inference::forward_pass);

}
