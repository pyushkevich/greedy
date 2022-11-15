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
#ifndef DIFFERENTIABLESCALINGANDSQUARING_H
#define DIFFERENTIABLESCALINGANDSQUARING_H

#include "lddmm_data.h"

/**
 * A layer implementing composition of a displacement field with itself
 * with efficient backpropagation
 */
template <unsigned int VDim, typename TReal>
class DisplacementSelfCompositionLayer
{
public:
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef typename LDDMMType::VectorImageType VectorImageType;

  // Forward pass - compute the composition of u with itself
  void Forward(VectorImageType *u, VectorImageType *v);

  // Forward pass - compute the composition of u with itself
  void ForwardSingleThreaded(VectorImageType *u, VectorImageType *v);

  // Backward pass - Given u and the partial derivative of some objective function f,
  // with respect to v, D_v f, compute D_u f = (D_u v)(D_v f)
  void Backward(VectorImageType *u, VectorImageType *Dv_f, VectorImageType *Du_f);

  // Backward pass - Given u and the partial derivative of some objective function f,
  // with respect to v, D_v f, compute D_u f = (D_u v)(D_v f)
  void BackwardSingleThreaded(VectorImageType *u, VectorImageType *Dv_f, VectorImageType *Du_f);

  // Test
  static bool TestDerivatives(bool multi_threaded);
};

/**
 * An experimental scaling and squaring layer that can be
 * backpropagated, allowing the use of scaling and squaring
 * for direct optimization
 */
template <unsigned int VDim, typename TReal>
class ScalingAndSquaringLayer
{
public:
  typedef DisplacementSelfCompositionLayer<VDim, TReal> CompositionLayer;
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef typename LDDMMType::VectorImageType VectorImageType;

  // Initialize the layer
  ScalingAndSquaringLayer(VectorImageType *u, unsigned int n_steps = 6);

  // Forward pass - compute the composition of u with itself
  void Forward(VectorImageType *u, VectorImageType *v);

  // Backward pass - Given u and the partial derivative of some objective function f,
  // with respect to v, D_v f, compute D_u f = (D_u v)(D_v f)
  void Backward(VectorImageType *u, VectorImageType *Dv_f, VectorImageType *Du_f);

  // Test
  static bool TestDerivatives(bool multi_threaded);

protected:
  CompositionLayer m_CompositionLayer;

  // During the forward pass, we need to store the U at every stage
  std::vector<typename VectorImageType::Pointer> m_StepU;

  typename VectorImageType::Pointer m_WorkImage1, m_WorkImage2;
  unsigned int m_Steps;
};

#endif // DIFFERENTIABLESCALINGANDSQUARING_H
