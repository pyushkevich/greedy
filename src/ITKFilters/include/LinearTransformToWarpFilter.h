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

#ifndef LINEARTRANSFORMTOWARPFILTER_H
#define LINEARTRANSFORMTOWARPFILTER_H

#include <itkImageToImageFilter.h>
#include "lddmm_common.h"

/**
 * This class transforms a linear transform into a deformation field
 */
template <class TInputImage, class TDeformationField, class TTransform>
class LinearTransformToWarpFilter
    : public itk::ImageToImageFilter<TInputImage,TDeformationField>
{
public:

  /** Standard class typedefs. */
  typedef LinearTransformToWarpFilter<TInputImage,TDeformationField,TTransform> Self;
  typedef itk::ImageToImageFilter<TInputImage,TDeformationField>   Superclass;
  typedef itk::SmartPointer<Self>                                  Pointer;
  typedef itk::SmartPointer<const Self>                            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( LinearTransformToWarpFilter, ImageToImageFilter )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension );

  // Lots of typedefs
  typedef TInputImage                                 InputImageType;
  typedef TDeformationField                           DeformationFieldType;
  typedef typename InputImageType::RegionType         OutputImageRegionType;
  typedef typename InputImageType::PixelType          InputPixelType;
  typedef typename InputImageType::InternalPixelType  InputComponentType;
  typedef typename InputImageType::IndexType          IndexType;
  typedef typename InputImageType::IndexValueType     IndexValueType;
  typedef typename InputImageType::SizeType           SizeType;
  typedef typename InputImageType::SpacingType        SpacingType;
  typedef typename InputImageType::DirectionType      DirectionType;
  typedef typename DeformationFieldType::PixelType    DeformationVectorType;
  typedef itk::ImageBase<ImageDimension>              ImageBaseType;

  typedef TTransform                                  TransformType;

  /** Set the fixed image */
  itkNamedInputMacro(FixedImage, InputImageType, "Primary")

  /** Set the moving image */
  itkNamedInputMacro(MovingImage, InputImageType, "moving")

  /** Set the transform */
  itkSetObjectMacro(Transform, TransformType)

  /** Get the transform */
  itkGetObjectMacro(Transform, TransformType)

protected:

  LinearTransformToWarpFilter() {}
  virtual ~LinearTransformToWarpFilter() {}

  void DynamicThreadedGenerateData(const OutputImageRegionType& outputRegionForThread) ITK_OVERRIDE;

  typename TransformType::Pointer m_Transform;

private:
  LinearTransformToWarpFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

#ifndef ITK_MANUAL_INSTANTIATION
#include "LinearTransformToWarpFilter.txx"
#endif

#endif // LINEARTRANSFORMTOWARPFILTER_H

