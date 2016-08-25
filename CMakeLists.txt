CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(LDDMM_TOOL)

FIND_PACKAGE(ITK 3.20 REQUIRED)

INCLUDE(${ITK_USE_FILE})
INCLUDE_DIRECTORIES(${LDDMM_TOOL_SOURCE_DIR}/src)

SET(ITK_USE_FFTWF ON)
INCLUDE(CMake/FindFFTW.cmake)

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

SET(GREEDY_LIB_SRC
  src/lddmm_data.cxx
  src/GreedyAPI.cxx
  src/GreedyParameters.cxx
  src/CommandLineHelper.cxx
)

SET(LDDMM_SRC src/lddmm_main.cxx)

SET(GREEDY_SRC src/greedy_main.cxx)

ADD_LIBRARY(greedyapi ${GREEDY_LIB_SRC} ${HEADERS})
LINK_LIBRARIES(${ITK_LIBRARIES} ${FFTWF_LIB} ${FFTWF_THREADS_LIB})

ADD_EXECUTABLE(lddmm ${LDDMM_SRC})
TARGET_LINK_LIBRARIES(lddmm greedyapi)

ADD_EXECUTABLE(greedy ${GREEDY_SRC})
TARGET_LINK_LIBRARIES(greedy greedyapi)

ADD_EXECUTABLE(test_accum testing/src/TestOneDimensionalInPlaceAccumulateFilter.cxx)
