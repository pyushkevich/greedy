PROJECT(GREEDY_TOOL)

#--------------------------------------------------------------------------------
# MACROS
#--------------------------------------------------------------------------------
# Get today's date (see http://cmake.3232098.n2.nabble.com/How-to-get-the-current-date-td5776870.html)
MACRO (TODAY RESULT)
    IF (WIN32)
        EXECUTE_PROCESS(COMMAND "cmd" " /C date /T" OUTPUT_VARIABLE ${RESULT})
        string(REGEX REPLACE "(..)/(..)/(....).*" "\\1/\\2/\\3" ${RESULT} ${${RESULT}})
    ELSEIF(UNIX)
        EXECUTE_PROCESS(COMMAND "date" "+%b %d, %Y" OUTPUT_VARIABLE ${RESULT})
        string(REGEX REPLACE "(...) (..), (....).*" "\\1 \\2, \\3" ${RESULT} ${${RESULT}})
    ELSE (WIN32)
        MESSAGE(SEND_ERROR "date not implemented")
        SET(${RESULT} 000000)
    ENDIF (WIN32)
	string(REPLACE "\n" "" ${RESULT} ${${RESULT}})
	string(REPLACE " " "" ${RESULT} ${${RESULT}})
ENDMACRO (TODAY)

#--------------------------------------------------------------------------------
# CONFIG VARIABLES FOR BUILDING AS A SUBPROJECT
#--------------------------------------------------------------------------------
OPTION(GREEDY_TOOL_BUILD_AS_SUBPROJECT "Build greedy inside of another CMake build" OFF)
MARK_AS_ADVANCED(GREEDY_TOOL_BUILD_AS_SUBPROJECT)

IF(GREEDY_TOOL_BUILD_AS_SUBPROJECT)
  OPTION(GREEDY_TOOL_SUBPROJECT_BUILD_CLI "Build the executables" OFF)
  OPTION(GREEDY_TOOL_SUBPROJECT_INSTALL_CLI "Install the executables" OFF)
  SET(BUILD_CLI ${GREEDY_TOOL_SUBPROJECT_BUILD_CLI})
  SET(INSTALL_CLI ${GREEDY_TOOL_SUBPROJECT_INSTALL_CLI})
ELSE()
  SET(BUILD_CLI ON)
  SET(INSTALL_CLI ON)
ENDIF()

#--------------------------------------------------------------------------------
# MAIN STUFF
#--------------------------------------------------------------------------------

# Configure whether LDDMM and potential other experimental binaries are built
OPTION(BUILD_LDDMM "Build experimental lddmm implementation?" OFF)

# If we are building as a sub-project we skip the extra steps
IF(NOT GREEDY_TOOL_BUILD_AS_SUBPROJECT)

  # Stand-alone build stuff
  CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

  # ITK
  FIND_PACKAGE(ITK 4.8.2 REQUIRED)
  INCLUDE(${ITK_USE_FILE})

  # Advanced option to link with FFT 
  OPTION(USE_FFTW "Use features provided by the FFTW library, only for experimental LDDMM code" OFF)

  # Deal with FFTW - only used by experimental LDDMM code
  IF(USE_FFTW)
    SET(ITK_USE_FFTWF ON)
    SET(ITK_USE_FFTWD ON)
    INCLUDE(CMake/FindFFTW.cmake)
    ADD_DEFINITIONS(-D_LDDMM_FFT_)
  ENDIF(USE_FFTW)

  # Include C-test
  ENABLE_TESTING()
  INCLUDE(CTest)

ENDIF(NOT GREEDY_TOOL_BUILD_AS_SUBPROJECT)

# Include the header directories
SET(GREEDY_INCLUDE_DIRS
  ${GREEDY_TOOL_SOURCE_DIR}/src
  ${GREEDY_TOOL_SOURCE_DIR}/src/ITKFilters/include
)

# Define header files
SET(HEADERS
  src/ITKFilters/include/FastLinearInterpolator.h
  src/ITKFilters/include/FastWarpCompositeImageFilter.h
  src/ITKFilters/include/FastWarpCompositeImageFilter.txx
  src/ITKFilters/include/JacobianDeterminantImageFilter.h
  src/ITKFilters/include/JacobianDeterminantImageFilter.txx
  src/ITKFilters/include/MultiImageAffineMSDMetricFilter.h
  src/ITKFilters/include/MultiImageAffineMSDMetricFilter.txx
  src/ITKFilters/include/MultiComponentImageMetricBase.h
  src/ITKFilters/include/MultiComponentImageMetricBase.txx
  src/ITKFilters/include/MultiImageOpticalFlowImageFilter.h
  src/ITKFilters/include/MultiImageOpticalFlowImageFilter.txx
  src/ITKFilters/include/MultiComponentMutualInfoImageMetric.h
  src/ITKFilters/include/MultiComponentMutualInfoImageMetric.txx
  src/ITKFilters/include/MultiComponentNCCImageMetric.h
  src/ITKFilters/include/MultiComponentNCCImageMetric.txx
  src/ITKFilters/include/OneDimensionalInPlaceAccumulateFilter.h
  src/ITKFilters/include/OneDimensionalInPlaceAccumulateFilter.txx
  src/ITKFilters/include/SimpleWarpImageFilter.h
  src/ITKFilters/include/SimpleWarpImageFilter.txx
  src/ITKFilters/include/itkGaussianInterpolateImageFunction.h
  src/ITKFilters/include/itkOptVectorLinearInterpolateImageFunction.h
  src/ITKFilters/include/itkOptVectorLinearInterpolateImageFunction.txx
  src/lddmm_common.h
  src/lddmm_data.h
  src/GreedyAPI.h
  src/GreedyException.h
  src/GreedyParameters.h
  src/MultiImageRegistrationHelper.h
  src/CommandLineHelper.h
)

# Define greedy library files
SET(GREEDY_LIB_SRC
  src/lddmm_data.cxx
  src/GreedyAPI.cxx
  src/GreedyParameters.cxx
  src/MultiImageRegistrationHelper.cxx
)

SET(LDDMM_SRC src/lddmm_main.cxx)
SET(GREEDY_SRC 
  ${CMAKE_CURRENT_BINARY_DIR}/GreedyVersion.cxx
  src/greedy_main.cxx)

ADD_LIBRARY(greedyapi ${GREEDY_LIB_SRC} ${HEADERS})
TARGET_INCLUDE_DIRECTORIES(greedyapi PUBLIC ${GREEDY_INCLUDE_DIRS})


IF(BUILD_CLI)

  # Get today's date (see http://cmake.3232098.n2.nabble.com/How-to-get-the-current-date-td5776870.html)
  TODAY(GREEDY_VERSION_COMPILE_DATE)

  # Get the current git hash
  include(${GREEDY_TOOL_SOURCE_DIR}/CMake/rpavlik/GetGitRevisionDescription.cmake)
  get_git_head_revision(GIT_REFSPEC GREEDY_VERSION_GIT_SHA1)

  # Get the current git branch
  include(${GREEDY_TOOL_SOURCE_DIR}/CMake/GitBranch.cmake)
  get_git_branch(GREEDY_VERSION_GIT_BRANCH)
  get_git_commit_date(${GREEDY_VERSION_GIT_SHA1} GREEDY_VERSION_GIT_TIMESTAMP)

  # Print the Git information
  MESSAGE(STATUS "Greedy GIT Info:")
  MESSAGE(STATUS "  Branch : ${GREEDY_VERSION_GIT_BRANCH}")
  MESSAGE(STATUS "  SHA    : ${GREEDY_VERSION_GIT_SHA1}")
  MESSAGE(STATUS "  Date   : ${GREEDY_VERSION_GIT_TIMESTAMP}")

  # One of the files needs to be configured (to insert version info)
  CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GreedyVersion.cxx.in
    ${CMAKE_CURRENT_BINARY_DIR}/GreedyVersion.cxx @ONLY IMMEDIATE)

  # Build the greedy library
  ADD_EXECUTABLE(greedy ${GREEDY_SRC})
  TARGET_LINK_LIBRARIES(greedy greedyapi ${ITK_LIBRARIES} ${FFTWF_LIB} ${FFTWF_THREADS_LIB})
  TARGET_INCLUDE_DIRECTORIES(greedy PUBLIC ${GREEDY_INCLUDE_DIRS})

  ADD_EXECUTABLE(test_accum testing/src/TestOneDimensionalInPlaceAccumulateFilter.cxx)
  TARGET_LINK_LIBRARIES(test_accum ${ITK_LIBRARIES})
  TARGET_INCLUDE_DIRECTORIES(test_accum PUBLIC ${GREEDY_INCLUDE_DIRS})

  # Configure LDDMM only if requested
  IF(BUILD_LDDMM)
    ADD_EXECUTABLE(lddmm ${LDDMM_SRC})
    TARGET_LINK_LIBRARIES(lddmm greedyapi
      ${ITK_LIBRARIES} ${FFTWF_LIB}
      ${FFTWF_THREADS_LIB})
    TARGET_INCLUDE_DIRECTORIES(lddmm PUBLIC ${GREEDY_INCLUDE_DIRS})
  ENDIF(BUILD_LDDMM)

ENDIF(BUILD_CLI)

IF(INSTALL_CLI)

  # Set the installation destination
  IF(GREEDY_TOOL_SUBPROJECT_CLI_INSTALL_PATH)
    SET(CLI_INSTALL_PATH ${GREEDY_TOOL_SUBPROJECT_CLI_INSTALL_PATH})
  ELSE()
    SET(CLI_INSTALL_PATH "bin")
  ENDIF()

  INSTALL(TARGETS greedy DESTINATION ${CLI_INSTALL_PATH} COMPONENT Runtime)
  IF(BUILD_LDDMM)
    INSTALL(TARGETS lddmm DESTINATION ${CLI_INSTALL_PATH} COMPONENT Runtime)
  ENDIF()

ENDIF(INSTALL_CLI)
