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
# MAIN STUFF
#--------------------------------------------------------------------------------

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
    INCLUDE(CMake/FindFFTW.cmake)
    ADD_DEFINITIONS(-D_LDDMM_FFT_)
  ENDIF(USE_FFTW)

  # Get today's date (see http://cmake.3232098.n2.nabble.com/How-to-get-the-current-date-td5776870.html)
  TODAY(GREEDY_VERSION_COMPILE_DATE)

  # Get the current git hash
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/rpavlik/")
  include(GetGitRevisionDescription)
  get_git_head_revision(GIT_REFSPEC GREEDY_VERSION_GIT_SHA1)

  # Get the current git branch
  include(GitBranch)
  get_git_branch(GREEDY_VERSION_GIT_BRANCH)
  get_git_commit_date(${GREEDY_VERSION_GIT_SHA1} GREEDY_VERSION_GIT_TIMESTAMP)

  # Print the Git information
  MESSAGE(STATUS "GIT Info:")
  MESSAGE(STATUS "  Branch : ${GREEDY_VERSION_GIT_BRANCH}")
  MESSAGE(STATUS "  SHA    : ${GREEDY_VERSION_GIT_SHA1}")
  MESSAGE(STATUS "  Date   : ${GREEDY_VERSION_GIT_TIMESTAMP}")

  # One of the files needs to be configured (to insert version info)
  CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GreedyVersion.cxx.in
    ${CMAKE_CURRENT_BINARY_DIR}/GreedyVersion.cxx @ONLY IMMEDIATE)

ENDIF(NOT GREEDY_TOOL_BUILD_AS_SUBPROJECT)

# Include the header directories
INCLUDE_DIRECTORIES(
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

# The executables are only compiled when the software is built as its own project
IF(NOT GREEDY_TOOL_BUILD_AS_SUBPROJECT)

  ADD_EXECUTABLE(lddmm ${LDDMM_SRC})
  TARGET_LINK_LIBRARIES(lddmm greedyapi
    ${ITK_LIBRARIES} ${FFTWF_LIB}
    ${FFTWF_THREADS_LIB})

  ADD_EXECUTABLE(greedy ${GREEDY_SRC})
  TARGET_LINK_LIBRARIES(greedy greedyapi ${ITK_LIBRARIES})

  ADD_EXECUTABLE(test_accum testing/src/TestOneDimensionalInPlaceAccumulateFilter.cxx)
  TARGET_LINK_LIBRARIES(test_accum ${ITK_LIBRARIES})

  ENABLE_TESTING()
  INCLUDE(CTest)

ENDIF(NOT GREEDY_TOOL_BUILD_AS_SUBPROJECT)
