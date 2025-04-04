CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
PROJECT(PICSL_Greedy)

# Enable languages
ENABLE_LANGUAGE(CXX)

# Specify the C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Avoid -fPIC errors
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Option, whether to fetch external dependencies or to find locally
set(FETCH_DEPENDENCIES OFF CACHE BOOL "Fetch dependencies (ITK, VTK and Greedy)?")
set(DEPENDENCIES_ONLY OFF CACHE BOOL "Only build dependencies, skip main build")

# If requested, fetch the dependencies
IF(FETCH_DEPENDENCIES)

    # Dependency: ITK
    include(FetchContent)
    set(Module_MorphologicalContourInterpolation ON CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        ITK
        GIT_REPOSITORY https://github.com/InsightSoftwareConsortium/ITK.git
        GIT_TAG        v5.4.0
    )
    FetchContent_MakeAvailable(ITK)
    SET(ITK_DIR ${itk_BINARY_DIR})
    FIND_PACKAGE(ITK CONFIG REQUIRED)
    INCLUDE(${ITK_USE_FILE})

    # ITK seems to mess things up by setting these variables
    UNSET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    UNSET(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    UNSET(CMAKE_LIBRARY_OUTPUT_DIRECTORY)

    # Dependency: VTK
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(VTK_REQUIRED_OBJCXX_FLAGS "" CACHE STRING "" FORCE)
    set(VTK_MODULE_ENABLE_VTK_IOHDF5 OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        VTK
        GIT_REPOSITORY https://github.com/Kitware/VTK.git
        GIT_TAG        v9.3.1
    )
    FetchContent_MakeAvailable(VTK)
    SET(VTK_DIR ${vtk_BINARY_DIR})
    FIND_PACKAGE(VTK REQUIRED COMPONENTS CommonCore IOCore IOLegacy IOPLY IOGeometry FiltersModeling)
    
    # Dependency: Greedy
    SET(GREEDY_BUILD_LDDMM ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        Greedy
        GIT_REPOSITORY https://github.com/pyushkevich/greedy.git
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(Greedy)
    # INCLUDE_DIRECTORIES(${greedy_SOURCE_DIR}/src)
    INCLUDE_DIRECTORIES(..)
    
ELSE()

    # Find ITK and C3D
    MESSAGE(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
    FIND_PACKAGE(ITK REQUIRED)
    INCLUDE(${ITK_USE_FILE})

    FIND_PACKAGE(VTK REQUIRED COMPONENTS CommonCore IOCore IOLegacy IOPLY IOGeometry FiltersModeling)

    FIND_PACKAGE(Greedy REQUIRED)

ENDIF()

# Set up VTK
vtk_module_autoinit(
  TARGETS greedyapi lmshootapi
  MODULES ${VTK_LIBRARIES})

# Python bindings
IF(NOT DEPENDENCIES_ONLY)
    SET(PYBIND11_FINDPYTHON ON)
    ADD_SUBDIRECTORY(external/pybind11)
    INCLUDE_DIRECTORIES(external/pybind11)
    pybind11_add_module(picsl_greedy GreedyPythonBindings.cxx)
    TARGET_INCLUDE_DIRECTORIES(picsl_greedy PRIVATE ${Greedy_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(picsl_greedy PUBLIC ${ITK_LIBRARIES} ${VTK_LIBRARIES} greedyapi lmshootapi multichunkgreedyapi)
    install(TARGETS picsl_greedy DESTINATION .)
ENDIF()
