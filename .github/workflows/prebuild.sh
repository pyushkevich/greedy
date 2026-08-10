#!/bin/bash
set -x -e 

pip install --upgrade pip setuptools wheel cmake ninja

echo "Check CMAKE version"
cmake --version

mkdir -p be/install && cd be

echo "Checking folder structure"
ls -lh .
ls -lh ..

# Download and build VTK
LIB_LOCATION=build
if [[ $1 =~ ubuntu-.* ]]; then
  # Kitware publishes the VTK wheel-SDK for x86_64 only — there is no linux aarch64
  # build at any released version — so on aarch64 we build VTK from source instead.
  # Note this branch previously matched on OS alone, so an arm runner would have
  # downloaded the x86_64 SDK.
  if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
    VTK_FROM_SOURCE=1
  else
    VTK_BINARY=vtk-wheel-sdk-9.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.tar.xz
  fi
  DYLD_SUFFIX=so
  MAKEFLAGS="-- -j 8"
elif [[ $1 == macos-13 ]]; then
  VTK_BINARY=vtk-wheel-sdk-9.3.1-cp310-cp310-macosx_10_10_x86_64.tar.xz
  DYLD_SUFFIX=dylib
  MAKEFLAGS="-- -j 8"
elif [[ $1 == macos-14 ]]; then
  VTK_BINARY=vtk-wheel-sdk-9.3.1-cp310-cp310-macosx_11_0_arm64.tar.xz
  DYLD_SUFFIX=dylib
elif [[ $1 =~ windows-.* ]]; then
  VTK_BINARY=vtk-wheel-sdk-9.3.1-cp310-cp310-win_amd64.tar.xz
  DYLD_SUFFIX=dll
  LIB_LOCATION=bin
  CMAKE_RELEASE_COMMAND="--config Release"
else
  exit 255
fi

# Install Eigen
git clone -b 3.4.0 https://gitlab.com/libeigen/eigen.git
cmake \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    -B eigen/build \
    eigen

cmake --build eigen/build --target install $MAKEFLAGS $CMAKE_RELEASE_COMMAND

mkdir -p install/vtk install/vtk/shared
if [[ -n "${VTK_FROM_SOURCE:-}" ]]; then
  # Build only the modules greedy's CMakeLists actually requires. Refusing the
  # Rendering/Qt/Views/Web groups is what keeps this cheap: the trimmed build takes
  # a couple of minutes, where a full VTK build would dominate the job.
  git clone --depth 1 -b v9.3.1 https://github.com/Kitware/VTK.git VTK
  cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_TESTING=OFF \
      -DVTK_BUILD_TESTING=OFF \
      -DVTK_BUILD_EXAMPLES=OFF \
      -DVTK_BUILD_DOCUMENTATION=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DVTK_GROUP_ENABLE_Rendering=DONT_WANT \
      -DVTK_GROUP_ENABLE_Qt=DONT_WANT \
      -DVTK_GROUP_ENABLE_Views=DONT_WANT \
      -DVTK_GROUP_ENABLE_Web=DONT_WANT \
      -DVTK_GROUP_ENABLE_Imaging=DONT_WANT \
      -DVTK_GROUP_ENABLE_MPI=DONT_WANT \
      -DVTK_GROUP_ENABLE_StandAlone=DONT_WANT \
      -DVTK_MODULE_ENABLE_VTK_CommonCore=YES \
      -DVTK_MODULE_ENABLE_VTK_IOCore=YES \
      -DVTK_MODULE_ENABLE_VTK_IOLegacy=YES \
      -DVTK_MODULE_ENABLE_VTK_IOPLY=YES \
      -DVTK_MODULE_ENABLE_VTK_IOGeometry=YES \
      -DVTK_MODULE_ENABLE_VTK_IOImage=YES \
      -DVTK_MODULE_ENABLE_VTK_IOXML=YES \
      -DVTK_MODULE_ENABLE_VTK_FiltersCore=YES \
      -DVTK_MODULE_ENABLE_VTK_FiltersGeneral=YES \
      -DVTK_MODULE_ENABLE_VTK_FiltersModeling=YES \
      -DCMAKE_INSTALL_PREFIX=$PWD/install/vtk \
      -B VTK/build \
      VTK
  cmake --build VTK/build --target install $MAKEFLAGS $CMAKE_RELEASE_COMMAND

  # Expose the config at the same path the wheel-SDK provides, so CIBW_ENVIRONMENT's
  # VTK_DIR needs no per-architecture variant. VTK's exported targets resolve library
  # paths RELATIVE to the config file — CMake walks three levels up from
  # vtk-9.3.1.data/headers/cmake and expects the prefix there — which is why VTK is
  # installed into install/vtk above rather than install/. Installing elsewhere makes
  # configure fail with "imported target references a file that does not exist".
  mkdir -p install/vtk/vtk-9.3.1.data/headers
  ln -sfn "$(dirname "$(find $PWD/install/vtk -name 'vtk-config.cmake' | head -1)")" \
          install/vtk/vtk-9.3.1.data/headers/cmake
  # Static build: nothing for auditwheel to bundle, so install/vtk/shared stays empty.
else
  # Install VTK from binary wheels provided by Kitware
  curl -L https://www.vtk.org/files/release/9.3/${VTK_BINARY} -o ./install/vtk/vtk-wheel-sdk.tar.xz
  tar -xJvf ./install/vtk/vtk-wheel-sdk.tar.xz --strip-components 1 -C $PWD/install/vtk
  ln $(find $PWD/install/vtk/${LIB_LOCATION} -name "*.${DYLD_SUFFIX}") install/vtk/shared
fi

# Link the shared libraries needed for delocate into a simple directory

# Build ITK
git clone -b v5.2.1 https://github.com/InsightSoftwareConsortium/ITK.git ITK
cmake \
    -DModule_MorphologicalContourInterpolation=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -B ITK/build \
    ITK

cmake --build ITK/build --target install $MAKEFLAGS $CMAKE_RELEASE_COMMAND

#git clone -b v9.3.1 https://github.com/Kitware/VTK.git VTK
#cmake \
#    -DBUILD_EXAMPLES=OFF \
#    -DBUILD_TESTING=OFF \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DBUILD_SHARED_LIBS=OFF \
#    -DVTK_REQUIRED_OBJCXX_FLAGS="" \
#    -DCMAKE_INSTALL_PREFIX=./install \
#    -B VTK/build \
#    VTK
#cmake --build VTK/build --target install --config Release

# Build Greedy
# git clone -b master https://github.com/pyushkevich/greedy.git greedy
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DGREEDY_BUILD_LMSHOOT=ON \
    -DGREEDY_BUILD_WRAPPING=ON \
    -DCMAKE_PREFIX_PATH="$PWD/install" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DVTK_DIR=$PWD/install/vtk/vtk-9.3.1.data/headers/cmake \
    -B greedy/build \
    ..

cmake --build greedy/build --target install $MAKEFLAGS $CMAKE_RELEASE_COMMAND
