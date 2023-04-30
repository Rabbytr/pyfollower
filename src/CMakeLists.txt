#
# src/CMakeLists.txt
# RVO2 Library
#
# Copyright 2023 Rabbytr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


add_subdirectory(ORCA)
add_subdirectory(pybind11)
include_directories(ORCA)

pybind11_add_module(${PROJECT_NAME} MODULE follower.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE RVO)
