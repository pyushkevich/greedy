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
#ifndef MULTICOMPONENTWeightedNCCIMAGEMETRIC_H
#define MULTICOMPONENTWeightedNCCIMAGEMETRIC_H

#include "MultiComponentImageMetricBase.h"

/**
 * Normalized cross-correlation metric. This filter sets up a mini-pipeline with
 * a pre-compute filter that interpolates the moving image, N one-dimensional
 * mean filters, and a post-compute filter that generates the metric and the
 * gradient.
 */
template <class TMetricTraits>
class ITK_EXPORT MultiComponentWeightedNCCImageMetric :
    public MultiComponentImageMetricBase<TMetricTraits>
{
public:
  /** Standard class typedefs. */
  typedef MultiComponentWeightedNCCImageMetric<TMetricTraits> Self;
  typedef MultiComponentImageMetricBase<TMetricTraits>        Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiComponentWeightedNCCImageMetric, MultiComponentImageMetricBase )

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename Superclass::InputImageType                InputImageType;
  typedef typename Superclass::InputPixelType                InputPixelType;
  typedef typename Superclass::InputComponentType            InputComponentType;
  typedef typename Superclass::MetricImageType               MetricImageType;
  typedef typename Superclass::MetricPixelType               MetricPixelType;
  typedef typename Superclass::GradientImageType             GradientImageType;
  typedef typename Superclass::GradientPixelType             GradientPixelType;
  typedef typename Superclass::MaskImageType                 MaskImageType;
  typedef typename Superclass::WeightVectorType              WeightVectorType;

  typedef typename Superclass::IndexType                     IndexType;
  typedef typename Superclass::IndexValueType                IndexValueType;
  typedef typename Superclass::SizeType                      SizeType;
  typedef typename Superclass::SpacingType                   SpacingType;
  typedef typename Superclass::DirectionType                 DirectionType;
  typedef typename Superclass::ImageBaseType                 ImageBaseType;

  /** Information from the deformation field class */
  typedef typename Superclass::DeformationFieldType          DeformationFieldType;

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, InputImageType::ImageDimension );

  /** Set the radius of the cross-correlation */
  itkSetMacro(Radius, SizeType)

  /** Get the radius of the cross-correlation */
  itkGetMacro(Radius, SizeType)

  /**
   * Set the working memory image for this filter. This function should be used to prevent
   * repeated allocation of memory when the metric is created/destructed in a loop. The
   * user can just pass in a pointer to a blank image, the filter will take care of allocating
   * the image as necessary
   */
  itkSetObjectMacro(WorkingImage, InputImageType)

  /**
   * Whether we should reuse the fixed components in the working image. This is true
   * if the filter is being run repeatedly on the same image
   */
  itkSetMacro(ReuseWorkingImageFixedComponents, bool)

  /**
   * Scaling of partial patches. When a fraction f of a patch covers the moving image, the NCC for that
   * patch would be scaled by f^k, where k is the weight scaling exponent. Default is 2. Only relevant
   * is Weighted is true.
   */
  itkSetMacro(WeightScalingExponent, unsigned int);
  itkGetMacro(WeightScalingExponent, unsigned int);

  /**
   * Get the gradient scaling factor. To get the actual gradient of the metric, multiply the
   * gradient output of this filter by the scaling factor. Explanation: for efficiency, the
   * metrics return an arbitrarily scaled vector, such that adding the gradient to the
   * deformation field would INCREASE SIMILARITY. For metrics that are meant to be minimized,
   * this is the opposite of the gradient direction. For metrics that are meant to be maximized,
   * it is the gradient direction.
   */
  virtual double GetGradientScalingFactor() const ITK_OVERRIDE { return 1.0; }

  /**
   * Implement a single non-threaded method and take care of threading internally
   */
  virtual void GenerateData() ITK_OVERRIDE;


protected:
  MultiComponentWeightedNCCImageMetric()
    : m_ReuseWorkingImageFixedComponents(false), m_WeightScalingExponent(2)
    { m_Radius.Fill(1); }

  ~MultiComponentWeightedNCCImageMetric() {}

  // TODO: set up for proper streaming
  // virtual void GenerateInputRequestedRegion();

  // Threaded method to compute accumulated components
  void PrecomputeAccumulatedComponents(const OutputImageRegionType &region);

  // Function to compute the metric and gradient at a pixel
  void ComputeNCCAndGradientAccumulatedComponents(const OutputImageRegionType &outputRegionForThread);

  // Function to compute the metric and gradient at a pixel
  void ComputeNCCGradient(const OutputImageRegionType &outputRegionForThread);

  // Accumulate some components of the working image (i.e., running sums)
  void AccumulateWorkingImageComponents(unsigned int comp_begin, unsigned int comp_end);

private:
  MultiComponentWeightedNCCImageMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // A pointer to the working image. The user should supply this image in order to prevent
  // unnecessary memory allocation
  typename InputImageType::Pointer m_WorkingImage;

  // Whether we should reuse the fixed components in the working image. This is true
  // if the filter is being run repeatedly on the same image
  bool m_ReuseWorkingImageFixedComponents;

  // Are we computing gradient
  bool m_NeedGradient;

  // Common values across the functions called internally
  unsigned int m_InputComponents;
  unsigned int m_FirstPassAccumComponents, m_FirstPassSavedComponents;
  unsigned int m_SecondPassAccumComponents;
  unsigned int m_SavedComponentsOffset;
  unsigned int m_TotalWorkingImageComponents;
  unsigned int m_WeightScalingExponent;

  // Iterator types
  typedef itk::ImageLinearConstIteratorWithIndex<InputImageType> InputIteratorTypeBase;
  typedef IteratorExtender<InputIteratorTypeBase> InputIteratorType;

  // Radius of the cross-correlation
  SizeType m_Radius;
};


#ifndef ITK_MANUAL_INSTANTIATION
#include "MultiComponentWeightedNCCImageMetric.txx"
#endif


#endif // MULTICOMPONENTWeightedNCCIMAGEMETRIC_H
