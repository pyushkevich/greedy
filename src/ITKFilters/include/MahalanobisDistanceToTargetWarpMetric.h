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

#ifndef MAHALANOBISDISTANCETOTARGETWARPMETRIC_H
#define MAHALANOBISDISTANCETOTARGETWARPMETRIC_H

#include "MultiComponentImageMetricBase.h"
#include <queue>
#include <vector>
#include <functional>

/**
 * Default traits for parameterizing the metric filters below.
 */
template <class TReal, unsigned int VDim>
struct DefaultMahalanobisDistanceToTargetMetricTraits
{
  typedef itk::VectorImage<TReal, VDim> InputImageType;
  typedef itk::Image<TReal, VDim> ScalarImageType;
  typedef itk::Image<itk::CovariantVector<TReal, VDim>, VDim> VectorImageType;

  typedef ScalarImageType MaskImageType;
  typedef VectorImageType DeformationFieldType;
  typedef VectorImageType GradientImageType;
  typedef ScalarImageType MetricImageType;
  typedef itk::MatrixOffsetTransformBase<TReal, VDim, VDim> TransformType;

  typedef TReal RealType;
};

/**
 * This metric computes the Mahalanobis distance between the displacement field 
 * and a multivariate Gaussian density on displacement at every voxel. The Gaussian
 * density is passed in as the fixed image, and encoded in a multi-component image
 * that at each voxel stores the mean displacement and the upper triangle and diagonal
 * of the inverse covariance matrix. The moving image is ignored.
 */
template <class TMetricTraits>
class ITK_EXPORT MahalanobisDistanceToTargetWarpMetric :
    public MultiComponentImageMetricBase<TMetricTraits>
{
public:
  /** Standard class typedefs. */
  typedef MahalanobisDistanceToTargetWarpMetric<TMetricTraits> Self;
  typedef MultiComponentImageMetricBase<TMetricTraits>         Superclass;
  typedef itk::SmartPointer<Self>                              Pointer;
  typedef itk::SmartPointer<const Self>                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MahalanobisDistanceToTargetWarpMetric, MultiComponentImageMetricBase )

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename Superclass::InputImageType                InputImageType;
  typedef typename Superclass::InputPixelType                InputPixelType;
  typedef typename Superclass::InputComponentType            InputComponentType;
  typedef typename Superclass::MetricImageType               MetricImageType;
  typedef typename Superclass::GradientImageType             GradientImageType;
  typedef typename Superclass::MetricPixelType               MetricPixelType;
  typedef typename Superclass::GradientPixelType             GradientPixelType;
  typedef typename Superclass::RealType                      RealType;

  typedef typename Superclass::IndexType                     IndexType;
  typedef typename Superclass::IndexValueType                IndexValueType;
  typedef typename Superclass::SizeType                      SizeType;
  typedef typename Superclass::SpacingType                   SpacingType;
  typedef typename Superclass::DirectionType                 DirectionType;
  typedef typename Superclass::ImageBaseType                 ImageBaseType;

  /** Information from the deformation field class */
  typedef typename Superclass::DeformationFieldType          DeformationFieldType;
  typedef typename Superclass::DeformationFieldPointer       DeformationFieldPointer;
  typedef typename Superclass::DeformationVectorType         DeformationVectorType;
  typedef typename Superclass::TransformType                 TransformType;

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, InputImageType::ImageDimension );

  /**
   * Get the gradient scaling factor. To get the actual gradient of the metric, multiply the
   * gradient output of this filter by the scaling factor. Explanation: for efficiency, the
   * metrics return an arbitrarily scaled vector, such that adding the gradient to the
   * deformation field would INCREASE SIMILARITY. For metrics that are meant to be minimized,
   * this is the opposite of the gradient direction. For metrics that are meant to be maximized,
   * it is the gradient direction.
   */
  virtual double GetGradientScalingFactor() const ITK_OVERRIDE { return 2.0; }

protected:

  virtual void DynamicThreadedGenerateData(const OutputImageRegionType& outputRegionForThread) ITK_OVERRIDE;

protected:
  MahalanobisDistanceToTargetWarpMetric() {}

  ~MahalanobisDistanceToTargetWarpMetric() {}

private:
  MahalanobisDistanceToTargetWarpMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};


#ifndef ITK_MANUAL_INSTANTIATION
#include "MahalanobisDistanceToTargetWarpMetric.txx"
#endif


#endif // MAHALANOBISDISTANCETOTARGETWARPMETRIC_H
