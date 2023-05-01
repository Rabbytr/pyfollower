/*
 * src/pyfollower.cpp
 *
 * Copyright 2023 Rabbytr
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <RVO.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "Agent.h"
// #include "pybind11/eigen.h"
#include "Vector2.h"

namespace py = pybind11;
using namespace py::literals;
using namespace RVO;

typedef RVOSimulator Simulator;

//std::vector<size_t> RVO::Simulator::pyGetAgentNeighbors(size_t agentNo) const {
//    std::vector<size_t> ret;
//    auto neigbors = agents_[agentNo]->agentNeighbors_;
//    ret.reserve(neigbors.size());
//    for (auto nbrs : neigbors) {
//        ret.emplace_back(nbrs.second->id_);
//    }
//    return ret;
//}
//
//size_t addObstacles(RVO::Simulator& self, const std::vector<std::tuple<float, float>> nodes) {
//    std::vector<RVO::Vector2> vertices;
//    vertices.reserve(nodes.size());
//    for (std::tuple<float, float> tp : nodes) {
//        vertices.emplace_back(RVO::Vector2(std::get<0>(tp), std::get<1>(tp)));
//    }
//    return self.addObstacle(vertices);
//}

py::array_t<double> getAgentPositions(const Simulator& self) {
    const int agent_number = static_cast<int>(self.getNumAgents());
    py::array_t<double> ret = py::array_t<double>(self.getNumAgents() * 2);
    ret.resize({ agent_number, 2 });
    auto r = ret.mutable_unchecked<2>();
    for (int i = 0; i < agent_number; i++) {
        const Vector2 position = self.getAgentPosition(i);
        r(i, 0) = position.x();
        r(i, 1) = position.y();
    }
    return ret;
}

py::array_t<double> getAgentVelocities(const Simulator& self) {
    const int agent_number = static_cast<int>(self.getNumAgents());
    py::array_t<double> ret = py::array_t<double>(self.getNumAgents() * 2);
    ret.resize({ agent_number, 2 });
    auto r = ret.mutable_unchecked<2>();
    for (int i = 0; i < agent_number; i++) {
        const Vector2 velocity = self.getAgentVelocity(i);
        r(i, 0) = velocity.x();
        r(i, 1) = velocity.y();
    }
    return ret;
}

int func_arg(const Simulator& self, const std::function<int(int)>& f) {
    return f(9);
}

PYBIND11_MODULE(follower, m) {
    py::class_<Simulator>(m, "Engine")
        .def(py::init<>())
        .def("set_agent_defaults", &Simulator::setAgentDefaults)
        .def("pref_velocity_correction", &Simulator::setPrefVelocityCorrection, "correction"_a)
        .def("set_timestep", &Simulator::setTimeStep, "timestep"_a, "Set the simulator timestep")
        .def("add_agent", static_cast<size_t(Simulator::*)(const Vector2 & position)>
            (&Simulator::addAgent), "position"_a)
        .def("follower_step", &Simulator::doStep, 
            py::call_guard<py::gil_scoped_release>())  // release the python GIL
        .def("set_agent_pref", &Simulator::setAgentPrefVelocity, "agent_id"_a, "pref"_a)
        .def("get_agent_position", &Simulator::getAgentPosition, "agent_id"_a)
        .def("get_agent_velocity", &Simulator::getAgentVelocity, "agent_id"_a)
        .def("get_agent_positions", &getAgentPositions)
        .def("get_agent_velocities", &getAgentVelocities)
        .def("get_global_time", &Simulator::getGlobalTime)
        .def("set_pvc_axis_function", &Simulator::setPvcAxisFunction, "pvc_axis_func"_a)

        .def("add_obstacles",&Simulator::addObstacle, "vertices"_a)
        .def("process_obstacles", &Simulator::processObstacles)
        //.def("get_agent_neigbors", &Simulator::pyGetAgentNeighbors, "agent_id"_a)
        .def("query_visibility", &Simulator::queryVisibility,"point1"_a, "point2"_a, "radius"_a)
        ;

    py::class_<Vector2>(m, "Vector2")
        .def(py::init<float, float>())
        .def("x", &Vector2::x)
        .def("y", &Vector2::y)
        .def("__repr__", [](const Vector2& self) {
            return "(" + std::to_string(self.x()) + ", " + std::to_string(self.y()) + ")";
        });
}