cmake_minimum_required(VERSION 2.8)
project(MCFHMM)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

# Find LibXml2 library
find_package(LibXml2 2.6.0 REQUIRED)

# Find Armadillo library
find_package(Armadillo REQUIRED)

# The ML_Pack library.
# Store the mlpack include directory in MLPACK_INCLUDE_DIR.
# The PATHS variable may be specified to give hints for where to find core.hpp.
find_path(MLPACK_INCLUDE_DIR
  NAMES core.hpp prereqs.hpp
  PATHS /usr/local/include/mlpack /usr/local/include/
)

if(MLPACK_INCLUDE_DIR)
	message(STATUS "Found MLPack include directory")
else()
	message(STATUS "ERROR: Could not find MLPack include directory")
endif()

# Find libmlpack.so (or equivalent) and store it in MLPACK_LIBRARY.
# If this example script were smarter, it would also find other dependencies of mlpack and store them in
# an MLPACK_LIBRARIES variable instead.
find_library(MLPACK_LIBRARY
  NAMES mlpack
  PATHS /usr/lib64/ /usr/lib/ /usr/local/lib64/ /usr/local/
)

if(MLPACK_LIBRARY)
	message(STATUS "Found MLPack link libraries")
else()
	message(STATUS "ERROR: Could not find MLPack link libraries")
endif()

# Define the framework files.
set(SRC
	src/main.cpp
	src/Sampler.cpp
	src/MCFHMM.cpp
	src/DETree.cpp
)

set(HDRs
	include/Sampler.h
	include/MCFHMM.h
	include/DETree.h
)

include_directories (
	include/
	${MLPACK_INCLUDE_DIR}
	${LIBXML2_INCLUDE_DIR}
	${ARMADILLO_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} ${SRC} ${HDRs})

target_link_libraries(${PROJECT_NAME} 
	${MLPACK_LIBRARY}
	${LIBXML2_LIBRARIES}
)
