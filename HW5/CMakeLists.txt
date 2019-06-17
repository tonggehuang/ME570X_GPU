# ME570X - Solid Modeling and GPU Computing
# MatrixMax

# CUDA_SELECT_NVCC_ARCH_FLAGS macro comes starting with CMake v3.7 series
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

# Project or solution name
project(matrixmax)

# Set CMake defaults
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH TRUE)

# Set install directory
set(APP_INSTALL_DIR ${PROJECT_BINARY_DIR}/install CACHE PATH "Application install directory" )
set(CMAKE_INSTALL_PREFIX ${APP_INSTALL_DIR})

# Add debug option
set(APP_ENABLE_DEBUG ON CACHE BOOL "Enable/disable debug mode")
if(APP_ENABLE_DEBUG)
  add_definitions(-DDEBUG)
endif()

# Silence MSVC warnings regarding to strdup, fopen, etc.
if(MSVC)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
endif()

# Set warning levels for host compilation
if(UNIX)
  add_compile_options(-Wno-unused-function)
  add_compile_options(-Wno-unused-parameter)
  add_compile_options(-Wextra)
  add_compile_options(-Wno-unused-variable)
endif()

# Activate GCC C++11 support for CMake versions before v3.1
if( CMAKE_VERSION VERSION_LESS 3.1 )
  message( STATUS "Using CMake " ${CMAKE_VERSION} )
  if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
    if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7 )
      message( FATAL_ERROR "You need at least GCC v4.7" )
    elseif( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0 )
      # Enable experimental C++11 support in GCC
      add_compile_options( -std=c++11 )
    else( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7 )
      # We are good to go!
    endif( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7 )
  endif( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
endif( CMAKE_VERSION VERSION_LESS 3.1 )

# Find necessary libraries
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
  message(FATAL_ERROR "nVidia CUDA is required for this software!")
endif()

# Add include directories of these libraries to the project
include_directories(${CUDA_INCLUDE_DIRS}})

# Workaround for "memcpy was not declared in this scope" when using GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
    if(CUDA_VERSION VERSION_LESS "8.0")
      message(STATUS "Found gcc >=5 and CUDA <= 7.5, adding workaround C++ flags")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
    endif(CUDA_VERSION VERSION_LESS "8.0")
  endif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

set(CUDA_PROPAGATE_HOST_FLAGS TRUE)

# For some MPs it may be necessary to specify the compute capability of your
# NVIDIA GPU. In that case, simply uncomment the following two lines that start
# with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
# hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
# or '21'; Kepler-class '30', '32', '35'; or Maxwell-class '50' or '52'
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.0 3.5+PTX 5.0 Maxwell 6.1)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

# Project source files
set(PROJECT_SOURCE_FILES
  MatrixMax.cu
)

# Create project
cuda_add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_FILES})

# Add debug suffix to the libraries
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

# Add an install target
install(
  TARGETS ${PROJECT_NAME}
  DESTINATION ${APP_INSTALL_DIR}
)
