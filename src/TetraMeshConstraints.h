#ifndef TETRAMESHCONSTRAINTS_H
#define TETRAMESHCONSTRAINTS_H

/*=========================================================================

  Program:   GreedyReg fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  GreedyReg initial development was funded by the NIH grant R01 EB017255.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

=========================================================================*/

class vtkUnstructuredGrid;
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_matrix_fixed.h>
#include <vnl/vnl_vector_fixed.h>
#include <vector>
#include <random>
#include "lddmm_data.h"
#include <vtkSmartPointer.h>

/**
 * A 'network' layer that takes as input the coordinates of tetraheadron and returns as output
 * the volume of this tetrahedron, with option to backpropagate.
 */
template <unsigned int VDim>
class TetraVolumeLayer
{
public:

  // Set the index of the vertex
  void SetIndex(const vnl_vector<int> v_index);

  // Forward pass, computes volume and the partial derivatives at each vertex
  double Forward(const vnl_matrix<double> &x, bool need_grad);

  // Backward pass
  void Backward(double d_tetra_vol, vnl_matrix<double> &d_x);

protected:
  vnl_vector<int> v_index;
  vnl_matrix_fixed<double, VDim+1, VDim> dv_dx;
};

/**
 * This class enables regularization terms based on tetrahedral (3D) or triangular (2D)
 * meshes, such as a penalty on squared difference in Jacobian determinant between
 * adjacent meshes
 */
template <class TFloat, unsigned int VDim>
class TetraMeshConstraints
{
public:
  typedef LDDMMData<TFloat, VDim> LDDMMType;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::ImageBaseType ImageBaseType;

  // Fixed size matrices
  typedef vnl_matrix_fixed<double, VDim, VDim> Mat;
  typedef vnl_vector_fixed<double, VDim> Vec;

  TetraMeshConstraints();

  // Initialize the mesh
  void SetMesh(vtkUnstructuredGrid *mesh);

  // Set the reference image
  void SetReferenceImage(ImageBaseType *ref_space);

  // Compute initial quantities - should be run after changing mesh or ref space
  void Initialize();

  // Compute the regularization term and its gradient with respect to a set of per-vertex
  // displacements specified in RAS coordinates
  double ComputeObjectiveAndGradientDisp(const vnl_matrix<double> &disp_ras, vnl_matrix<double> &grad, double weight);

  // Compute the regularization term and its gradient with respect to the voxel-space
  // displacement field phi
  double ComputeObjectiveAndGradientPhi(VectorImageType *phi_vox, VectorImageType *grad, double weight = 1.0);

  // A function to test derivatives on dummy data or real data
  static bool TestDerivatives(std::mt19937 &rnd, ImageBaseType *refspace = nullptr, vtkUnstructuredGrid *mesh = nullptr);

protected:
  // The mesh object
  vtkSmartPointer<vtkUnstructuredGrid> m_MeshVTK;

  // The reference space
  typename ImageBaseType::Pointer m_Reference;

  // List of tetrahedral vertex indices extracted from the mesh, N x (VDim+1) array
  vnl_matrix<int> m_TetraVI;

  // List of all adjacent tetrahedra, N x 2 array
  std::vector< std::pair<int, int> > m_TetraNbr;

  // List of all tetrahedral vertex coordinates
  vnl_matrix<double> m_TetraX_Vox, m_TetraX_RAS;
  vnl_matrix<double> m_TetraX_RAS_Disp, m_TetraX_RAS_Warped, m_D_TetraX_RAS_Warped;

  // Network layer used for volume computation and backprop
  std::vector< TetraVolumeLayer<VDim> > m_TetraVolumeLayer;

  // Volumes of tetrahedra in fixed image
  vnl_vector<double> m_TetraVol, m_TetraVol_Warped, m_D_TetraVol_Warped;

  // Transform between RAS to voxel space
  Mat A_vox_to_ras, A_ras_to_vox;
  Vec b_vox_to_ras, b_ras_to_vox;
};

#endif // TETRAMESHCONSTRAINTS_H
