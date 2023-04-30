#
# CMakeLists.txt
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

cmake_minimum_required( VERSION 3.10 )
project( follower )
set (CMAKE_CXX_STANDARD 17)

# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

# === Use multithreading?

option(FOLLOWER_USE_MULTITHREADING "Allow the program to use multiple parallel threads" ON)
if (FOLLOWER_USE_MULTITHREADING)
	if (MSVC)
	  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /openmp")
	else()
	  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif ()
endif (FOLLOWER_USE_MULTITHREADING)

# === Linux compatibility compiler option

if (NOT MSVC)
	add_definitions(-O3 -lstdc++fs -fPIC -shared)
endif() 

SET(CMAKE_BUILD_TYPE "Release")

add_subdirectory(src)
