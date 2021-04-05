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
#ifndef AFFINECOSTFUNCTIONS_H
#define AFFINECOSTFUNCTIONS_H

#include <vnl/vnl_cost_function.h>
#include <vnl/vnl_random.h>
#include <vnl/vnl_trace.h>
#include <lddmm_data.h>
#include <deque>

namespace itk {
  template <typename T, unsigned int D1, unsigned int D2> class MatrixOffsetTransformBase;
}

template <unsigned int VDim, typename TReal> class GreedyApproach;
template <typename T, unsigned int V> class MultiImageOpticalFlowHelper;
class GreedyParameters;

/**
 * Parent of all affine/rigid cost functions
 */
template <unsigned int VDim, typename TReal = double>
class AbstractAffineCostFunction : public vnl_cost_function
{
public:
  typedef itk::MatrixOffsetTransformBase<TReal, VDim, VDim> LinearTransformType;
  typedef MultiImageOpticalFlowHelper<TReal, VDim> OFHelperType;
  typedef GreedyApproach<VDim, TReal> ParentType;

  // Image type definitions
  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef typename LDDMMType::ImageType ImageType;
  typedef typename LDDMMType::ImagePointer ImagePointer;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::VectorImagePointer VectorImagePointer;

  AbstractAffineCostFunction(int n_unknowns) : vnl_cost_function(n_unknowns) {}
  virtual vnl_vector<double> GetCoefficients(LinearTransformType *tran) = 0;
  virtual void GetTransform(const vnl_vector<double> &coeff, LinearTransformType *tran) = 0;
  virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g);
  virtual ImageType *GetMetricImage() = 0;

  // This is the full compute method, which returns both the average metric over the
  // masked region and the mask size, allowing multiple registrations to be combined
  // in a way that is weighted by the mask
  virtual void ComputeWithMask(vnl_vector<double> const& x,
                               double *f_metric, vnl_vector<double>* g_metric,
                               double *f_mask, vnl_vector<double>* g_mask) = 0;
};

class LineSearchMemory
{
public:
  struct Entry {
    vnl_vector<double> g, x;
    double f;
  };

  void update(vnl_vector<double> const& x, double f, vnl_vector<double>* g);

protected:
  std::deque<Entry> data;
};

/**
 * Pure affine cost function - parameters are elements of N x N matrix M.
 * Transformation takes place in voxel coordinates - not physical coordinates (for speed)
 */
template <unsigned int VDim, typename TReal = double>
class PureAffineCostFunction : public AbstractAffineCostFunction<VDim, TReal>
{
public:
  typedef AbstractAffineCostFunction<VDim, TReal> Superclass;
  typedef typename Superclass::ParentType ParentType;
  typedef typename Superclass::OFHelperType OFHelperType;
  typedef typename Superclass::LinearTransformType LinearTransformType;

  typedef typename Superclass::ImageType ImageType;
  typedef typename Superclass::ImagePointer ImagePointer;
  typedef typename Superclass::VectorImageType VectorImageType;
  typedef typename Superclass::VectorImagePointer VectorImagePointer;

  // Construct the function
  PureAffineCostFunction(GreedyParameters *param, ParentType *parent,
                         unsigned int group, unsigned int level,
                         OFHelperType *helper);

  // Get the parameters for the specified initial transform
  vnl_vector<double> GetCoefficients(LinearTransformType *tran) override;

  // Get the transform for the specificed coefficients
  void GetTransform(const vnl_vector<double> &coeff, LinearTransformType *tran) override;

  // Get the preferred scaling for this function given image dimensions
  virtual vnl_vector<double> GetOptimalParameterScaling(const itk::Size<VDim> &image_dim);

  // Compute cost function and gradient, along with the mask volume and gradient
  virtual void ComputeWithMask(vnl_vector<double> const& x,
                               double *f_metric, vnl_vector<double>* g_metric,
                               double *f_mask, vnl_vector<double>* g_mask) override;

  // Get the metric image
  virtual ImageType *GetMetricImage() override { return m_Metric; }

protected:

  // Data needed to compute the cost function
  GreedyParameters *m_Param;
  OFHelperType *m_OFHelper;
  GreedyApproach<VDim, TReal> *m_Parent;
  bool m_Allocated;
  unsigned int m_Group, m_Level;

  // Storage for the gradient of the similarity map
  ImagePointer m_Metric;

  // Last set of coefficients evaluated
  vnl_vector<double> last_coeff;

  // For debugging line searches
  LineSearchMemory m_LineSearchMemory;
};

/**
 * Physical space affine cost function - parameters are elements of affine transform in
 * physical RAS space.
 */
template <unsigned int VDim, typename TReal = double>
class PhysicalSpaceAffineCostFunction : public AbstractAffineCostFunction<VDim, TReal>
{
public:
  typedef AbstractAffineCostFunction<VDim, TReal> Superclass;
  typedef typename Superclass::ParentType ParentType;
  typedef typename Superclass::OFHelperType OFHelperType;
  typedef typename Superclass::LinearTransformType LinearTransformType;

  typedef typename Superclass::ImageType ImageType;
  typedef typename Superclass::ImagePointer ImagePointer;
  typedef typename Superclass::VectorImageType VectorImageType;
  typedef typename Superclass::VectorImagePointer VectorImagePointer;

  PhysicalSpaceAffineCostFunction(GreedyParameters *param, ParentType *parent,
                                  unsigned int group, unsigned int level,
                                  OFHelperType *helper);
  virtual vnl_vector<double> GetCoefficients(LinearTransformType *tran) override;
  virtual void GetTransform(const vnl_vector<double> &coeff, LinearTransformType *tran) override;
  virtual vnl_vector<double> GetOptimalParameterScaling(const itk::Size<VDim> &image_dim);

  // Compute cost function and gradient, along with the mask volume and gradient
  virtual void ComputeWithMask(vnl_vector<double> const& x,
                               double *f_metric, vnl_vector<double>* g_metric,
                               double *f_mask, vnl_vector<double>* g_mask) override;

  void map_phys_to_vox(const vnl_vector<double> &x_phys, vnl_vector<double> &x_vox);

  // Get the metric image
  virtual ImageType *GetMetricImage() override { return m_PureFunction.GetMetricImage(); }

protected:
  PureAffineCostFunction<VDim, TReal> m_PureFunction;

  // Voxel to physical transforms for fixed, moving image
  typedef vnl_matrix_fixed<double, VDim, VDim> Mat;
  typedef vnl_vector_fixed<double, VDim> Vec;

  Mat Q_fix, Q_mov, Q_fix_inv, Q_mov_inv;
  Vec b_fix, b_mov, b_fix_inv, b_mov_inv;

  vnl_matrix<double> J_phys_vox;
};

/**
 * Abstract scaling cost function - wraps around another cost function and provides scaling.
 *
 * Note: the scaling function takes over ownership of the wrapped function and will delete
 * the pointer to the wrapped function.
 */
template <unsigned int VDim, typename TReal = double>
class ScalingCostFunction : public AbstractAffineCostFunction<VDim, TReal>
{
public:
  typedef AbstractAffineCostFunction<VDim, TReal> Superclass;
  typedef typename Superclass::ParentType ParentType;
  typedef typename Superclass::OFHelperType OFHelperType;
  typedef typename Superclass::LinearTransformType LinearTransformType;

  typedef typename Superclass::ImageType ImageType;
  typedef typename Superclass::ImagePointer ImagePointer;
  typedef typename Superclass::VectorImageType VectorImageType;
  typedef typename Superclass::VectorImagePointer VectorImagePointer;

  // Construct the function
  ScalingCostFunction(Superclass *pure_function, const vnl_vector<double> &scaling)
    : Superclass(pure_function->get_number_of_unknowns()),
      m_PureFunction(pure_function), m_Scaling(scaling) {}

  ~ScalingCostFunction() { delete m_PureFunction; }

  // Get the parameters for the specified initial transform
  vnl_vector<double> GetCoefficients(LinearTransformType *tran) override;

  // Get the transform for the specificed coefficients
  void GetTransform(const vnl_vector<double> &coeff, LinearTransformType *tran) override;

  // Cost function computation
  virtual void ComputeWithMask(vnl_vector<double> const& x,
                               double *f_metric, vnl_vector<double>* g_metric,
                               double *f_mask, vnl_vector<double>* g_mask) override;

  const vnl_vector<double> &GetScaling() { return m_Scaling; }

  // Get the metric image
  virtual ImageType *GetMetricImage() override { return m_PureFunction->GetMetricImage(); }

protected:

  // Data needed to compute the cost function. We use std::shared_ptr here to avoid
  // the need for the caller to clean up the pure function
  Superclass *m_PureFunction;
  vnl_vector<double> m_Scaling;
};

/** A function that integrates affine metric across multiple image groups */
template <unsigned int VDim, typename TReal = double>
class MaskWeightedSumAffineConstFunction : public AbstractAffineCostFunction<VDim, TReal>
{
public:
  typedef AbstractAffineCostFunction<VDim, TReal> Superclass;
  typedef typename Superclass::ParentType ParentType;
  typedef typename Superclass::OFHelperType OFHelperType;
  typedef typename Superclass::LinearTransformType LinearTransformType;

  typedef typename Superclass::ImageType ImageType;
  typedef typename Superclass::ImagePointer ImagePointer;
  typedef typename Superclass::VectorImageType VectorImageType;
  typedef typename Superclass::VectorImagePointer VectorImagePointer;

  /** Create a function, giving it ownership of an array of component functions */
  MaskWeightedSumAffineConstFunction(std::vector<Superclass *> components)
    : Superclass(components.front()->get_number_of_unknowns()),
      m_Components(components) {}

  ~MaskWeightedSumAffineConstFunction();

  virtual void ComputeWithMask(vnl_vector<double> const& x,
                               double *f_metric, vnl_vector<double>* g_metric,
                               double *f_mask, vnl_vector<double>* g_mask) override;

  virtual vnl_vector<double> GetCoefficients(LinearTransformType *tran) override
    { return m_Components.front()->GetCoefficients(tran); }

  virtual void GetTransform(const vnl_vector<double> &coeff, LinearTransformType *tran) override
    { return m_Components.front()->GetTransform(coeff, tran); }

  // TODO: this is incorrect, but why do we need MetricImage?
  virtual ImageType *GetMetricImage() override
    { return m_Components.front()->GetMetricImage(); }

private:
  // Component functions
  std::vector<Superclass *> m_Components;
};

/** Cost function for rigid registration */
template <unsigned int VDim, typename TReal = double>
class RigidCostFunction : public AbstractAffineCostFunction<VDim, TReal>
{
public:
  typedef AbstractAffineCostFunction<VDim, TReal> Superclass;
  typedef typename Superclass::ParentType ParentType;
  typedef typename Superclass::OFHelperType OFHelperType;
  typedef typename Superclass::LinearTransformType LinearTransformType;

  typedef typename Superclass::ImageType ImageType;
  typedef typename Superclass::ImagePointer ImagePointer;
  typedef typename Superclass::VectorImageType VectorImageType;
  typedef typename Superclass::VectorImagePointer VectorImagePointer;


  typedef vnl_vector_fixed<double, VDim> Vec;
  typedef vnl_matrix_fixed<double, VDim, VDim> Mat;

  RigidCostFunction(GreedyParameters *param, ParentType *parent,
                    unsigned int group, unsigned int level, OFHelperType *helper);
  vnl_vector<double> GetCoefficients(LinearTransformType *tran) override;
  void GetTransform(const vnl_vector<double> &coeff, LinearTransformType *tran) override;
  virtual void ComputeWithMask(vnl_vector<double> const& x,
                               double *f_metric, vnl_vector<double>* g_metric,
                               double *f_mask, vnl_vector<double>* g_mask) override;

  // Get the preferred scaling for this function given image dimensions
  virtual vnl_vector<double> GetOptimalParameterScaling(const itk::Size<VDim> &image_dim);

  // Generate a random rotation matrix with rotation angle alpha (radians)
  static Mat GetRandomRotation(vnl_random &randy, double alpha);

  // Get the metric image
  virtual ImageType *GetMetricImage() override { return m_AffineFn.GetMetricImage(); }

protected:

  static Mat GetRotationMatrix(const Vec &q);
  static Vec GetAxisAngle(const Mat &R);

  // We wrap around a physical space affine function, since rigid in physical space is not
  // the same as rigid in voxel space
  PhysicalSpaceAffineCostFunction<VDim, TReal> m_AffineFn;

  // Flip matrix -- allows rigid registration with flips. If the input matrix has a flip,
  // that flip is maintained during registration
  Mat flip;

};


/** Specialized function for 2D rigid registration */
template <typename TReal>
class RigidCostFunction<2, TReal> : public AbstractAffineCostFunction<2, TReal>
{
public:
  typedef AbstractAffineCostFunction<2, TReal> Superclass;
  typedef typename Superclass::ParentType ParentType;
  typedef typename Superclass::OFHelperType OFHelperType;
  typedef typename Superclass::LinearTransformType LinearTransformType;

  typedef typename Superclass::ImageType ImageType;
  typedef typename Superclass::ImagePointer ImagePointer;
  typedef typename Superclass::VectorImageType VectorImageType;
  typedef typename Superclass::VectorImagePointer VectorImagePointer;

  const static int VDim = 2;
  typedef vnl_vector_fixed<double, VDim> Vec;
  typedef vnl_matrix_fixed<double, VDim, VDim> Mat;


  RigidCostFunction(GreedyParameters *param, ParentType *parent,
                    unsigned int group, unsigned int level, OFHelperType *helper);
  vnl_vector<double> GetCoefficients(LinearTransformType *tran) override;
  void GetTransform(const vnl_vector<double> &coeff, LinearTransformType *tran) override;
  virtual void ComputeWithMask(vnl_vector<double> const& x,
                               double *f_metric, vnl_vector<double>* g_metric,
                               double *f_mask, vnl_vector<double>* g_mask) override;

  // Get the preferred scaling for this function given image dimensions
  virtual vnl_vector<double> GetOptimalParameterScaling(const itk::Size<VDim> &image_dim);

  // Generate a random rotation matrix with rotation angle alpha (radians)
  static Mat GetRandomRotation(vnl_random &randy, double alpha);

  // Get the metric image
  virtual ImageType *GetMetricImage() override { return m_AffineFn.GetMetricImage(); }

protected:

  static Mat GetRotationMatrix(double theta);
  static double GetRotationAngle(const Mat &R);

  // We wrap around a physical space affine function, since rigid in physical space is not
  // the same as rigid in voxel space
  PhysicalSpaceAffineCostFunction<VDim, TReal> m_AffineFn;

  // Flip matrix -- allows rigid registration with flips. If the input matrix has a flip,
  // that flip is maintained during registration
  Mat flip;

};



#endif // AFFINECOSTFUNCTIONS_H
