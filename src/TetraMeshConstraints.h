#ifndef TETRAMESHCONSTRAINTS_H
#define TETRAMESHCONSTRAINTS_H

/*=========================================================================

  Program:   ALFABIS fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  This program is part of ALFABIS: Adaptive Large-Scale Framework for
  Automatic Biomedical Image Segmentation.

  ALFABIS development is funded by the NIH grant R01 EB017255.

  ALFABIS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ALFABIS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ALFABIS.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================*/

class vtkUnstructuredGrid;
#include <vnl_matrix.h>
#include <vector>

/**
 * This class enables regularization terms based on tetrahedral (3D) or triangular (2D)
 * meshes, such as a penalty on squared difference in Jacobian determinant between
 * adjacent meshes
 */
template <class TFloat, unsigned int VDim>
class TetraMeshConstraints
{
public:

  TetraMeshConstraints();

  // Initialize the mesh
  void SetMesh(vtkUnstructuredGrid *mesh);

protected:
  // The mesh object
  vtkUnstructuredGrid *m_MeshVTK;

  // List of tetrahedral vertex indices extracted from the mesh, N x (VDim+1) array
  vnl_matrix<int> m_TetraVI;

  // List of all adjacent tetrahedra, N x 2 array
  std::vector< std::pair<int, int> > m_TetraNbr;

  // List of all tetrahedral vertex coordinates
  vnl_matrix<double> m_TetraX;
};

#endif // TETRAMESHCONSTRAINTS_H
