PROJECT(GREEDY_TOOL)

# If we are building as a sub-project we skip the extra steps
IF(NOT GREEDY_BUILD_AS_SUBPROJECT)

  # Stand-alone build stuff
  CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

  # ITK
  FIND_PACKAGE(ITK 4.8.2 REQUIRED)
  INCLUDE(${ITK_USE_FILE})

  # Deal with FFTW - only used by experimental LDDMM code
  SET(ITK_USE_FFTWF ON)
  INCLUDE(CMake/FindFFTW.cmake)

ENDIF(NOT GREEDY_BUILD_AS_SUBPROJECT)

# Include the header directories
INCLUDE_DIRECTORIES(${GREEDY_TOOL_SOURCE_DIR}/src)

# Define header files
SET(HEADERS
  src/FastLinearInterpolator.h
  src/FastWarpCompositeImageFilter.h
  src/FastWarpCompositeImageFilter.txx
  src/LinearTransformToWarpFilter.h
  src/LinearTransformToWarpFilter.txx
  src/MultiImageAffineMSDMetricFilter.h
  src/MultiImageAffineMSDMetricFilter.txx
  src/MultiComponentImageMetricBase.h
  src/MultiComponentImageMetricBase.txx
  src/MultiImageOpticalFlowImageFilter.h
  src/MultiImageOpticalFlowImageFilter.txx
  src/MultiImageRegistrationHelper.h
  src/MultiImageRegistrationHelper.txx
  src/MultiComponentMutualInfoImageMetric.h
  src/MultiComponentMutualInfoImageMetric.txx
  src/MultiComponentNCCImageMetric.h
  src/MultiComponentNCCImageMetric.txx
  src/OneDimensionalInPlaceAccumulateFilter.h
  src/OneDimensionalInPlaceAccumulateFilter.txx
  src/SimpleWarpImageFilter.h
  src/SimpleWarpImageFilter.txx
  src/itkGaussianInterpolateImageFunction.h
  src/itkOptVectorLinearInterpolateImageFunction.h
  src/itkOptVectorLinearInterpolateImageFunction.txx
  src/lddmm_common.h
  src/lddmm_data.h
  src/GreedyAPI.h
  src/GreedyParameters.h
  src/CommandLineHelper.h
)

# Define greedy library files
SET(GREEDY_LIB_SRC
  src/lddmm_data.cxx
  src/GreedyAPI.cxx
  src/GreedyParameters.cxx
  src/CommandLineHelper.cxx
)

SET(LDDMM_SRC src/lddmm_main.cxx)
SET(GREEDY_SRC src/greedy_main.cxx)

ADD_LIBRARY(greedyapi ${GREEDY_LIB_SRC} ${HEADERS})

ADD_EXECUTABLE(lddmm ${LDDMM_SRC})
TARGET_LINK_LIBRARIES(lddmm greedyapi 
  ${ITK_LIBRARIES} ${FFTWF_LIB}
  ${FFTWF_THREADS_LIB})

ADD_EXECUTABLE(greedy ${GREEDY_SRC})
TARGET_LINK_LIBRARIES(greedy greedyapi ${ITK_LIBRARIES})

ADD_EXECUTABLE(test_accum
  testing/src/TestOneDimensionalInPlaceAccumulateFilter.cxx)
TARGET_LINK_LIBRARIES(test_accum ${ITK_LIBRARIES})
