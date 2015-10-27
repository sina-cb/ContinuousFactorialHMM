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
)

add_executable(${PROJECT_NAME} ${SRC} ${HDRs})

#find_library(FANN_LIB fann)
#target_link_libraries (
#Box2D
#glui
#}
