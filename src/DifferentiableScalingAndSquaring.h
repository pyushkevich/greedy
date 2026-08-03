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
#ifndef DIFFERENTIABLESCALINGANDSQUARING_H
#define DIFFERENTIABLESCALINGANDSQUARING_H

#include "lddmm_data.h"
#include <functional>
#include <deque>

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

  // Helper methods for testing
  static typename VectorImageType::Pointer MakeTestDisplacement(
      int size = 96, TReal scale = 8.0, TReal sigma = 1.0, bool orient_ras = false);

  // Test
  static bool TestDerivatives();
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
  typedef typename LDDMMType::Vec Vec;


  // Initialize the layer
  ScalingAndSquaringLayer(VectorImageType *u, unsigned int n_steps = 6);

  // Forward pass - compute the composition of u with itself
  void Forward(VectorImageType *u, VectorImageType *v);

  // Backward pass - Given u and the partial derivative of some objective function f,
  // with respect to v, D_v f, compute D_u f = (D_u v)(D_v f)
  void Backward(VectorImageType *u, VectorImageType *Dv_f, VectorImageType *Du_f);

  // Test
  static bool TestDerivatives(double noise_amplitude = 8.0, double noise_sigma = 1.0);

protected:
  CompositionLayer m_CompositionLayer;

  // During the forward pass, we need to store the U at every stage
  std::vector<typename VectorImageType::Pointer> m_WorkImage;
  unsigned int m_Steps;
};

/**
 * A penalty on the derivative of the stationary velocity field, used to
 * impose smoothness constraints during optimization
 */
template <unsigned int VDim, typename TReal>
class DisplacementFieldSmoothnessLoss
{
public:
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::Vec Vec;

  double ComputeLossAndGradient(VectorImageType *u, VectorImageType *grad, TReal grad_scale = 1.0);

  // Test
  static bool TestDerivatives();

protected:

};

/**
 * Adam step implementation
 */
template <typename TImage>
class AdamStep
{
public:
  AdamStep(double alpha = 0.001, double beta_1 = 0.9, double beta_2 = 0.999, double eps = 1.0e-8)
    : alpha(alpha), beta_1(beta_1), beta_2(beta_2), eps(eps) {};

  void Compute(int iter, const TImage *gradient, TImage *m_k, TImage *v_k, TImage *theta);

protected:
  double alpha, beta_1, beta_2, eps;
};

/**
 * LBFGS implementation - modeled on Pytorch but works with images
 */
template <unsigned int VDim, typename TReal>
class ImageLBFGS
{
public:

  // Optimization is performed over deformation fields and such - which are of type VectorImage
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::VectorImagePointer VectorImagePointer;

  ImageLBFGS(double lr = 1.0, double tolerance_grad=1e-07, double tolerance_change=1e-09,
             int history_size=10, bool strong_wolfe = false);

  /**
   * Typedef of the closure that must be passed to the step function. Parameters are x, grad
   */
  typedef std::function<double(const VectorImageType *, VectorImageType *)> Closure;

  /**
   * Perform a step, updating the x vector, passing in the storage for the gradient. The memory
   * for history will be allocated by the class. Returns true if convergence criteria are met.
   */
  bool Step(Closure closure, VectorImageType *x, double &obj, VectorImageType *grad);

protected:
  double lr, tolerance_grad, tolerance_change;
  int history_size = 10;
  bool strong_wolfe;
  int n_iter = 0;

  // History storage
  std::deque<VectorImagePointer> s_i, y_i;

  // Scalar history
  std::deque<double> rho_i;

  // Alphas and betas
  std::vector<double> alpha, beta;

  // Last gradient computed
  VectorImagePointer g_last;

  // Last update direction
  VectorImagePointer d;

  // Hessian scalar value
  double H_diag;

  // Last step size
  double t;

  // Rotate history
  VectorImagePointer rotate_history(std::deque<VectorImagePointer> &hist, VectorImageType *ref);
};

#endif // DIFFERENTIABLESCALINGANDSQUARING_H
