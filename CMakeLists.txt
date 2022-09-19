PROJECT(GREEDY)

# Required CMake version
CMAKE_MINIMUM_REQUIRED(VERSION 3.10.2)
if(COMMAND cmake_policy)
  CMAKE_POLICY(SET CMP0003 NEW)
  CMAKE_POLICY(SET CMP0077 NEW)
  CMAKE_POLICY(SET CMP0042 NEW)
endif(COMMAND cmake_policy)

# C++11 required
set (CMAKE_CXX_STANDARD 11)

#--------------------------------------------------------------------------------
# VERSION
#--------------------------------------------------------------------------------
INCLUDE(CMake/PYVersion.cmake)

# Set the semantic version and version date
VERSION_VARS(1 2 0 "" "20200112" "Jan 12, 2021")

#--------------------------------------------------------------------------------
# Build configuration
#--------------------------------------------------------------------------------
INCLUDE(CMakeDependentOption)

# Subproject build options
OPTION(GREEDY_BUILD_AS_SUBPROJECT "Build greedy inside of another CMake build" OFF)
MARK_AS_ADVANCED(GREEDY_BUILD_AS_SUBPROJECT)
IF(GREEDY_BUILD_AS_SUBPROJECT)
  OPTION(GREEDY_SUBPROJECT_BUILD_CLI "Build the executables" OFF)
  OPTION(GREEDY_SUBPROJECT_INSTALL_CLI "Install the executables" OFF)
  SET(BUILD_CLI ${GREEDY_SUBPROJECT_BUILD_CLI})
  SET(INSTALL_CLI ${GREEDY_SUBPROJECT_INSTALL_CLI})
ELSE()
  SET(BUILD_CLI ON)
  SET(INSTALL_CLI ON)
ENDIF()

# Do we want to build LDDMM?
OPTION(GREEDY_BUILD_LDDMM "Build experimental LDDMM implementation" OFF)

# Do we want to use FFTW?
CMAKE_DEPENDENT_OPTION(GREEDY_USE_FFTW 
  "Use features provided by the FFTW library, only for experimental LDDMM code" OFF
  "GREEDY_BUILD_LDDMM" OFF)

# Do we want to build MACF tools?
OPTION(GREEDY_BUILD_MACF "Build experimental multi-atlas correspondence fusion code" OFF)

# Do we want to build the stack registration binary
OPTION(GREEDY_BUILD_STACK_REG "Build experimental slice stack registration code" OFF)

# Do we want to enable PDE-based methods that require sparse solvers
OPTION(GREEDY_USE_SPARSE_SOLVERS "Build registration tools that use sparse solvers" OFF)

#--------------------------------------------------------------------------------
# Dependent packages
#--------------------------------------------------------------------------------

# ITK
FIND_PACKAGE(ITK 5.1.2 REQUIRED)
INCLUDE(${ITK_USE_FILE})

# VTK - required for quality mesh transformations
FIND_PACKAGE(VTK 9.1.0 REQUIRED COMPONENTS
  CommonCore
  IOCore
  IOLegacy
  IOPLY
  IOGeometry
  IOImage
  FiltersCore
  FiltersGeneral)
SET(GREEDY_VTK_LIBRARIES ${VTK_LIBRARIES})

# Deal with FFTW - only used by experimental LDDMM code
IF(GREEDY_USE_FFTW)
  SET(ITK_USE_FFTWF ON)
  SET(ITK_USE_FFTWD ON)
  INCLUDE(CMake/FindFFTW.cmake)
  ADD_DEFINITIONS(-D_LDDMM_FFT_)
ENDIF()

# Deal with sparse solvers
IF(GREEDY_USE_SPARSE_SOLVERS)
  ADD_DEFINITIONS(-D_LDDMM_SPARSE_SOLVERS_)
  ADD_SUBDIRECTORY(src/SparseSolvers)
  INCLUDE_DIRECTORIES(src/SparseSolvers)
  SET(SPARSE_LIBRARY sparsesolvers)
ENDIF()

# Include the header directories
INCLUDE_DIRECTORIES(
  ${GREEDY_SOURCE_DIR}/src
  ${GREEDY_SOURCE_DIR}/src/ITKFilters/include
)

#--------------------------------------------------------------------------------
# Hard-coded documentation
#--------------------------------------------------------------------------------

# Markdown documentation compiled into the C code
# modified from: https://github.com/starseeker/tinyscheme-cmake/blob/master/CMakeLists.txt
# # Rather than load the init.scm file at run time,
# # with the uncertainties as to where exactly the file
# # resides, use William Ahern's hexdump to generate
# # an embeddable version. Build our own copy of hexdump
# # to ensure consistent behavior and portability.
# # See http://25thandclement.com/~william/projects/hexdump.c.html
ADD_EXECUTABLE(text_to_hex utilities/hexdump.c)
SET_PROPERTY(TARGET text_to_hex APPEND PROPERTY COMPILE_DEFINITIONS "HEXDUMP_MAIN")

# List of docs that we want to generate
SET(HARDCODED_DOCS
  stackg_usage_main stackg_usage_init stackg_usage_recon
  stackg_usage_volmatch stackg_usage_voladd stackg_usage_voliter
  stackg_usage_splat)

UNSET(HARDCODED_DOCS_HEADERS)
FOREACH(doc ${HARDCODED_DOCS})

  ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${doc}.h
    COMMAND text_to_hex -i ${CMAKE_CURRENT_SOURCE_DIR}/docs/usage/${doc}.txt > ${CMAKE_CURRENT_BINARY_DIR}/${doc}.h
    DEPENDS text_to_hex ${CMAKE_CURRENT_SOURCE_DIR}/docs/usage/${doc}.txt)

  SET(HARDCODED_DOCS_HEADERS ${HARDCODED_DOCS_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/${doc}.h)

ENDFOREACH()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

ADD_CUSTOM_TARGET(docs_to_hex ALL DEPENDS ${HARDCODED_DOCS_HEADERS})

#--------------------------------------------------------------------------------
# Source file specification
#--------------------------------------------------------------------------------

# One of the files needs to be configured (to insert version info)
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GreedyVersion.cxx.in
  ${CMAKE_CURRENT_BINARY_DIR}/GreedyVersion.cxx @ONLY IMMEDIATE)

# Define header files
SET(HEADERS
  src/ITKFilters/include/CompositeImageNanMaskingFilter.h
  src/ITKFilters/include/CompositeImageNanMaskingFilter.txx
  src/ITKFilters/include/FastLinearInterpolator.h
  src/ITKFilters/include/FastWarpCompositeImageFilter.h
  src/ITKFilters/include/FastWarpCompositeImageFilter.txx
  src/ITKFilters/include/JacobianDeterminantImageFilter.h
  src/ITKFilters/include/JacobianDeterminantImageFilter.txx
  src/ITKFilters/include/MultiComponentImageMetricBase.h
  src/ITKFilters/include/MultiComponentImageMetricBase.txx
  src/ITKFilters/include/MultiComponentMetricReport.h
  src/ITKFilters/include/MultiComponentMutualInfoImageMetric.h
  src/ITKFilters/include/MultiComponentMutualInfoImageMetric.txx
  src/ITKFilters/include/MultiComponentNCCImageMetric.h
  src/ITKFilters/include/MultiComponentNCCImageMetric.txx
  src/ITKFilters/include/MultiComponentWeightedNCCImageMetric.h
  src/ITKFilters/include/MultiComponentWeightedNCCImageMetric.txx
  src/ITKFilters/include/MultiImageOpticalFlowImageFilter.h
  src/ITKFilters/include/MultiImageOpticalFlowImageFilter.txx
  src/ITKFilters/include/OneDimensionalInPlaceAccumulateFilter.h
  src/ITKFilters/include/OneDimensionalInPlaceAccumulateFilter.txx
  src/ITKFilters/include/itkSLICSuperVoxelImageFilter.h
  src/ITKFilters/include/itkSLICSuperVoxelImageFilter.hxx
  src/lddmm_common.h
  src/lddmm_data.h
  src/AffineCostFunctions.h
  src/AffineTransformUtilities.h
  src/GreedyAPI.h
  src/GreedyMeshIO.h
  src/GreedyException.h
  src/GreedyParameters.h
  src/MultiImageRegistrationHelper.h
  src/CommandLineHelper.h
)

# Define greedy library files
SET(GREEDY_LIB_SRC
  src/lddmm_data.cxx
  src/lddmm_sparse.cxx
  src/GreedyAPI.cxx
  src/GreedyMeshIO.cxx
  src/GreedyParameters.cxx
  src/MultiImageRegistrationHelper.cxx
  src/AffineCostFunctions.cxx
)

SET(LDDMM_SRC src/lddmm_main.cxx)
SET(GREEDY_SRC 
  ${CMAKE_CURRENT_BINARY_DIR}/GreedyVersion.cxx
  src/greedy_main.cxx)

SET(MACF_SRC src/macf_optimization.cxx)

SET(STACK_GREEDY_SRC
  ${CMAKE_CURRENT_BINARY_DIR}/GreedyVersion.cxx
  src/dijkstra/ShortestPath.cxx
  src/greedy_stack_reg.cxx)

SET(CHUNK_GREEDY_SRC
  ${CMAKE_CURRENT_BINARY_DIR}/GreedyVersion.cxx
  testing/src/greedy_multi_chunk.cxx)

SET(TEMPLATE_AVG_SRC
  ${CMAKE_CURRENT_BINARY_DIR}/GreedyVersion.cxx
  src/template_make_average.cxx)

ADD_LIBRARY(greedyapi ${GREEDY_LIB_SRC} ${HEADERS})
TARGET_INCLUDE_DIRECTORIES(greedyapi PUBLIC ${GREEDY_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(greedyapi ${VTK_LIBRARIES} ${ITK_LIBRARIES})

vtk_module_autoinit(
  TARGETS greedyapi
  MODULES ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS greedy
  MODULES ${VTK_LIBRARIES})

# Variable referencing greedyapi and libs it depends on
SET(GREEDY_API_LIBS greedyapi
  ${ITK_LIBRARIES} ${GREEDY_VTK_LIBRARIES}
  ${FFTWF_LIB} ${FFTWF_THREADS_LIB}
  ${SPARSE_LIBRARY})

# propagation api
SET(PROPAGATION_HDR
  src/Propagation/PropagationAPI.h
  src/Propagation/PropagationCommon.h
  src/Propagation/PropagationTools.h
  src/Propagation/PropagationData.h
  src/Propagation/PropagationIO.h
)

SET(PROPAGATION_SRC
  src/Propagation/PropagationAPI.cxx
  src/Propagation/PropagationData.txx
  src/Propagation/PropagationTools.txx
  src/Propagation/PropagationIO.cxx
)

ADD_LIBRARY(PropagationAPI ${PROPAGATION_SRC})
TARGET_INCLUDE_DIRECTORIES(PropagationAPI PUBLIC ${GREEDY_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(PropagationAPI ${GREEDY_API_LIBS})

# List of installable targets
SET(GREEDY_INSTALL_TARGETS greedy greedy_template_average multi_chunk_greedy)

# The executables are only compiled when the software is built as its own project
IF(BUILD_CLI)
  IF(GREEDY_BUILD_LDDMM)
    ADD_EXECUTABLE(lddmm ${LDDMM_SRC})
    SET(GREEDY_INSTALL_TARGETS ${GREEDY_INSTALL_TARGETS} lddmm)
    TARGET_LINK_LIBRARIES(lddmm ${GREEDY_API_LIBS})
  ENDIF()

  IF(GREEDY_BUILD_MACF)
    ADD_EXECUTABLE(macf ${MACF_SRC})
    SET(GREEDY_INSTALL_TARGETS ${GREEDY_INSTALL_TARGETS} macf)
    TARGET_LINK_LIBRARIES(macf ${GREEDY_API_LIBS})
  ENDIF()

  IF(GREEDY_BUILD_STACK_REG)
    ADD_EXECUTABLE(stack_greedy ${STACK_GREEDY_SRC})
    SET(GREEDY_INSTALL_TARGETS ${GREEDY_INSTALL_TARGETS} stack_greedy)
    INCLUDE_DIRECTORIES(${GREEDY_SOURCE_DIR}/src/dijkstra)
    TARGET_LINK_LIBRARIES(stack_greedy ${GREEDY_API_LIBS})
    ADD_DEPENDENCIES(stack_greedy docs_to_hex)
  ENDIF()

  ADD_EXECUTABLE(greedy ${GREEDY_SRC})
  TARGET_LINK_LIBRARIES(greedy ${GREEDY_API_LIBS} ${FFTWF_LIB} ${FFTWF_THREADS_LIB} ${SPARSE_LIBRARY} PropagationAPI)
  TARGET_INCLUDE_DIRECTORIES(greedy PUBLIC ${GREEDY_SOURCE_DIR}/src/propagation)

  ADD_EXECUTABLE(greedy_template_average ${TEMPLATE_AVG_SRC})
  TARGET_LINK_LIBRARIES(greedy_template_average ${GREEDY_API_LIBS})

  ADD_EXECUTABLE(test_accum testing/src/TestOneDimensionalInPlaceAccumulateFilter.cxx)
  TARGET_LINK_LIBRARIES(test_accum ${ITK_LIBRARIES})

  ADD_EXECUTABLE(test_greedy testing/src/GreedyTestDriver.cxx)
  TARGET_LINK_LIBRARIES(test_greedy ${GREEDY_API_LIBS})

  ADD_EXECUTABLE(multi_chunk_greedy ${CHUNK_GREEDY_SRC})
  TARGET_LINK_LIBRARIES(multi_chunk_greedy ${GREEDY_API_LIBS})
ENDIF(BUILD_CLI)

# Install command-line executables
IF(INSTALL_CLI)
  IF(GREEDY_SUBPROJECT_CLI_INSTALL_PATH)
    SET(CLI_INSTALL_PATH ${GREEDY_SUBPROJECT_CLI_INSTALL_PATH})
  ELSE()
    SET(CLI_INSTALL_PATH bin)
  ENDIF()

  INSTALL(TARGETS ${GREEDY_INSTALL_TARGETS} DESTINATION ${CLI_INSTALL_PATH} COMPONENT Runtime)

ENDIF(INSTALL_CLI)

IF(NOT GREEDY_BUILD_AS_SUBPROJECT)

  INCLUDE(CMake/Package.cmake)

  ENABLE_TESTING()
  INCLUDE(CTest)

  # Path to the test data
  SET(TESTING_DATADIR ${CMAKE_CURRENT_SOURCE_DIR}/testing/data)

  # Add NCC metric Matlab regression tests
  ADD_TEST(NAME "NCC_Matlab_Regression"  COMMAND test_greedy ncc_gradient_vs_matlab 0 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "WNCC_Matlab_Regression" COMMAND test_greedy ncc_gradient_vs_matlab 1 WORKING_DIRECTORY ${TESTING_DATADIR})

  # Add derivative tests
  ADD_TEST(NAME "grad_metric_phi_WNCC_nomask" COMMAND test_greedy grad_metric_phi 2 0.00001 0.0001 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -m WNCC 2x2 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "grad_metric_phi_NCC_nomask"  COMMAND test_greedy grad_metric_phi 2 0.00001 0.0001 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -m NCC 2x2 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "grad_metric_phi_SSD_nomask"  COMMAND test_greedy grad_metric_phi 2 0.00001 0.01 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -m SSD WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "grad_metric_phi_WNCC_bimask" COMMAND test_greedy grad_metric_phi 2 0.00001 0.0001 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -gm t1longi_2d_bl_mask.nii.gz -mm t1longi_2d_fu_mask.nii.gz -m WNCC 2x2 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "grad_metric_phi_NCC_bimask"  COMMAND test_greedy grad_metric_phi 2 0.00001 0.0001 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -gm t1longi_2d_bl_mask.nii.gz -mm t1longi_2d_fu_mask.nii.gz -m NCC 2x2 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "grad_metric_phi_SSD_bimask"  COMMAND test_greedy grad_metric_phi 2 0.00001 0.01 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -gm t1longi_2d_bl_mask.nii.gz -mm t1longi_2d_fu_mask.nii.gz -m SSD WORKING_DIRECTORY ${TESTING_DATADIR})

  ADD_TEST(NAME "grad_metric_aff_WNCC_nomask" COMMAND test_greedy grad_metric_aff 2 0.0000001 0.01 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -m WNCC 2x2 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "grad_metric_aff_NCC_nomask"  COMMAND test_greedy grad_metric_aff 2 0.0000001 0.01 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -m NCC 2x2 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "grad_metric_aff_SSD_nomask"  COMMAND test_greedy grad_metric_aff 2 0.0000001 0.01 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -m SSD WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "grad_metric_aff_WNCC_bimask" COMMAND test_greedy grad_metric_aff 2 0.0000001 0.01 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -gm t1longi_2d_bl_mask.nii.gz -mm t1longi_2d_fu_mask.nii.gz -m WNCC 2x2 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "grad_metric_aff_NCC_bimask"  COMMAND test_greedy grad_metric_aff 2 0.0000001 0.01 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -gm t1longi_2d_bl_mask.nii.gz -mm t1longi_2d_fu_mask.nii.gz -m NCC 2x2 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "grad_metric_aff_SSD_bimask"  COMMAND test_greedy grad_metric_aff 2 0.0000001 0.01 -i t1longi_2d_bl.nii.gz t1longi_2d_fu.nii.gz -ia t1longi_2d_rigid_halfway.mat -gm t1longi_2d_bl_mask.nii.gz -mm t1longi_2d_fu_mask.nii.gz -m SSD WORKING_DIRECTORY ${TESTING_DATADIR})

  # Tests for non-zero background images
  ADD_TEST(NAME "grad_metric_nzbkg_aff_SSD_nomask"  COMMAND test_greedy grad_metric_aff 2 0.0000001 0.01 -i circle_nzbkg_fixed.nii.gz circle_nzbkg_moving.nii.gz -ia circle_nzbkg_init.mat -bg 4 -m SSD WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "grad_metric_nzbkg_phi_SSD_nomask"  COMMAND test_greedy grad_metric_phi 2 0.0000001 0.01 -i circle_nzbkg_fixed.nii.gz circle_nzbkg_moving.nii.gz -ia circle_nzbkg_init.mat -bg 4 -m SSD WORKING_DIRECTORY ${TESTING_DATADIR})

  # Add the phantom series of tests
  ADD_TEST(NAME "Phantom_NMI_Affine_NoMask" COMMAND test_greedy phantom 1 2 NMI 12 0 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "Phantom_NCC_Affine_Mask"   COMMAND test_greedy phantom 1 3 NCC 12 1 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "Phantom_WNCC_Affine_Mask"  COMMAND test_greedy phantom 1 3 WNCC 12 1 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "Phantom_SSD_Rigid_NoMask"  COMMAND test_greedy phantom 1 1 SSD 6 0 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "Phantom_NMI_Rigid_NoMask"  COMMAND test_greedy phantom 1 2 NMI 6 0 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "Phantom_NCC_Rigid_NoMask"  COMMAND test_greedy phantom 1 3 NCC 6 0 WORKING_DIRECTORY ${TESTING_DATADIR})
  ADD_TEST(NAME "Phantom_WNCC_Rigid_NoMask" COMMAND test_greedy phantom 1 3 WNCC 6 0 WORKING_DIRECTORY ${TESTING_DATADIR})

ENDIF(NOT GREEDY_BUILD_AS_SUBPROJECT)
