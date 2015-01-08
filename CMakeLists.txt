CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(LDDMM_TOOL)

FIND_PACKAGE(ITK 3.20 REQUIRED)

INCLUDE(${ITK_USE_FILE})
INCLUDE_DIRECTORIES(${LDDMM_TOOL_SOURCE_DIR}/src)

SET(ITK_USE_FFTWF ON)
INCLUDE(CMake/FindFFTW.cmake)

SET(HEADERS
  src/FastLinearInterpolator.h
  src/MultiImageSimpleWarpImageFilter.h
  src/SimpleWarpImageFilter.h
  src/itkGaussianInterpolateImageFunction.h
  src/itkOptLinearInterpolateImageFunction.h
  src/itkOptLinearInterpolateImageFunction.txx
  src/itkOptVectorLinearInterpolateImageFunction.h
  src/itkOptVectorLinearInterpolateImageFunction.txx
  src/lddmm_common.h
  src/lddmm_data.h
  src/MultiImageSimpleWarpImageFilter.txx
  src/SimpleWarpImageFilter.txx
)

SET(LDDMM_SRC
  src/lddmm_main.cxx
  src/lddmm_data.cxx
  src/lddmm_temp.cxx)

SET(GREEDY_SRC
  src/greedy_main.cxx
  src/lddmm_data.cxx)

LINK_LIBRARIES(${ITK_LIBRARIES} ${FFTWF_LIB} ${FFTWF_THREADS_LIB})
ADD_EXECUTABLE(lddmm ${LDDMM_SRC} ${HEADERS})
ADD_EXECUTABLE(greedy ${GREEDY_SRC} ${HEADERS})
