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
#ifndef __MultiImageOpticalFlowImageFilter_h
#define __MultiImageOpticalFlowImageFilter_h

#include "MultiComponentImageMetricBase.h"


/**
 * \class MultiImageOpticalFlowImageFilter
 * \brief Warps an image using an input deformation field (for LDDMM)
 *
 * This filter efficiently computes the optical flow field between a
 * set of image pairs, given a transformation phi. This filter is the
 * workhorse of deformable and affine rigid registration algorithms that
 * use the mean squared difference metric. Given a set of fixed images F_i
 * and moving images M_i, it computes
 *
 *   v(x) = Sum_i w_i \[ F_i(x) - M_i(Phi(x)) ] \Grad M_i (Phi(x))
 *
 * The efficiency of this filter comes from combining the interpolation of
 * all the M and GradM terms in one loop, so that all possible computations
 * are reused
 *
 * The fixed and moving images must be passed in to the filter in the form
 * of VectorImages of size K and (VDim+K), respectively - i.e., the moving
 * images and their gradients are packed together.
 *
 * The output should be an image of CovariantVector type
 *
 * \warning This filter assumes that the input type, output type
 * and deformation field type all have the same number of dimensions.
 *
 */
template <class TMetricTraits>
class ITK_EXPORT MultiImageOpticalFlowImageFilter :
    public MultiComponentImageMetricBase<TMetricTraits>
{
public:
  /** Standard class typedefs. */
  typedef MultiImageOpticalFlowImageFilter<TMetricTraits>   Self;
  typedef MultiComponentImageMetricBase<TMetricTraits>      Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageOpticalFlowImageFilter, MultiComponentImageMetricBase )

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
  virtual double GetGradientScalingFactor() const override { return -2.0; }

  /**
   * What form of the gradient to return. The default is to return the gradient of the
   * objective function [F - M \circ u]. The alternative is to use the Demons-like form
   * in Eq.4 of Vercauteren 2008 NeuroImage "Diffeomorphic Demons: Efficient Non-parametric
   * Image Registration".
   *
   * u = \frac{F - M \circ \phi}{(|J|^2 + \sigma} J
   *
   * where J = \nabla ( M \circ \phi )
   */
  itkSetMacro(UseDemonsGradientForm, bool)
  itkGetMacro(UseDemonsGradientForm, bool)

  /**
   * The value of \frac{\sigma_i}{sigma_x^2} in Eq.4 of Vercauteren 2008 NeuroImage. The
   * default is set to 0.1 but is pretty meaningless as it depends on the norm of the
   * gradient of M
   */
  itkSetMacro(DemonsSigma, double)
  itkGetMacro(DemonsSigma, double)


protected:
  MultiImageOpticalFlowImageFilter() : m_UseDemonsGradientForm(false), m_DemonsSigma(0.1) {}
  ~MultiImageOpticalFlowImageFilter() {}

  void DynamicThreadedGenerateData(const OutputImageRegionType& outputRegionForThread) override;

private:
  MultiImageOpticalFlowImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool m_UseDemonsGradientForm;
  double m_DemonsSigma;
};












#ifndef ITK_MANUAL_INSTANTIATION
#include "MultiImageOpticalFlowImageFilter.txx"
#endif

#endif
