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

//std::vector<size_t> RVO::RVOSimulator::pyGetAgentNeighbors(size_t agentNo) const {
//    std::vector<size_t> ret;
//    auto neigbors = agents_[agentNo]->agentNeighbors_;
//    ret.reserve(neigbors.size());
//    for (auto nbrs : neigbors) {
//        ret.emplace_back(nbrs.second->id_);
//    }
//    return ret;
//}
//
//size_t addObstacles(RVO::RVOSimulator& self, const std::vector<std::tuple<float, float>> nodes) {
//    std::vector<RVO::Vector2> vertices;
//    vertices.reserve(nodes.size());
//    for (std::tuple<float, float> tp : nodes) {
//        vertices.emplace_back(RVO::Vector2(std::get<0>(tp), std::get<1>(tp)));
//    }
//    return self.addObstacle(vertices);
//}

py::array_t<double> getAgentPositions(const RVOSimulator& self) {
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

py::array_t<double> getAgentVelocities(const RVOSimulator& self) {
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

int func_arg(const RVOSimulator& self, const std::function<int(int)>& f) {
    return f(10);
}



PYBIND11_MODULE(follower, m) {
    py::class_<RVOSimulator>(m, "Engine")
        .def(py::init<>())
        .def("set_agent_defaults", &RVOSimulator::setAgentDefaults)
        .def("pref_velocity_correction", &RVOSimulator::setPrefVelocityCorrection, "correction"_a)
        .def("set_timestep", &RVOSimulator::setTimeStep, "timestep"_a, "Set the simulator timestep")
        .def("add_agent", static_cast<size_t(RVOSimulator::*)(const Vector2 & position)>
            (&RVOSimulator::addAgent), "position"_a)
        .def("set_agent_pref", &RVOSimulator::setAgentPrefVelocity, "agent_id"_a, "pref"_a)
        .def("follower_step", &RVOSimulator::doStep)
        .def("get_agent_position", &RVOSimulator::getAgentPosition, "agent_id"_a)
        .def("get_agent_velocity", &RVOSimulator::getAgentVelocity, "agent_id"_a)
        .def("get_agent_positions", &getAgentPositions)
        .def("get_agent_velocities", &getAgentVelocities)
        .def("get_global_time", &RVOSimulator::getGlobalTime)
        .def("func_test", &func_arg)

        .def("add_obstacles",&RVOSimulator::addObstacle, "vertices"_a)
        .def("process_obstacles", &RVOSimulator::processObstacles)
        //.def("get_agent_neigbors", &RVOSimulator::pyGetAgentNeighbors, "agent_id"_a)
        .def("query_visibility", &RVOSimulator::queryVisibility,"point1"_a, "point2"_a, "radius"_a)
        ;

    py::class_<Vector2>(m, "Vector2")
        .def(py::init<float, float>())
        .def("x", &Vector2::x)
        .def("y", &Vector2::y)
        .def("__repr__", [](const Vector2& self) {
            return "(" + std::to_string(self.x()) + ", " + std::to_string(self.y()) + ")";
        });
}