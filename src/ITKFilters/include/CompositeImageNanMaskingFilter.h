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
#ifndef COMPOSITEIMAGENANMASKINGFILTER_H
#define COMPOSITEIMAGENANMASKINGFILTER_H

#include "itkInPlaceImageFilter.h"
#include "itkVectorImage.h"

/**
 * This filter handles the mask-based and nan-based masking for Greedy.
 *
 * Its inputs are a VectorImage containing components of a fixed or moving
 * image to be included in registration and a fixed or moving image mask. The
 * VectorImage may contain NaNs. Its output (in place) is a VectorImage that
 * has been masked and with NaNs removed. Its secondary output is a mask that
 * incorporates the input mask as well as NaNs. At every pixel where one of the
 * input components is NaN, the mask will be set to zero and all components will
 * be set to zero as well.
 */
template <class TCompositeImage, class TMaskImage>
class CompositeImageNanMaskingFilter :
    public itk::InPlaceImageFilter<TCompositeImage, TCompositeImage>
{
public:
  // Image and region typedefs
  using InputImageType = TCompositeImage;
  using MaskImageType  = TMaskImage;
  using RegionType     = typename InputImageType::RegionType;

  // Standard ITK filter typedefs
  using Self         = CompositeImageNanMaskingFilter;
  using Superclass   = itk::InPlaceImageFilter<TCompositeImage, TCompositeImage>;
  using Pointer      = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using DataObjectPointer = typename itk::ProcessObject::DataObjectPointer;
  using DataObjectIdentifierType = typename itk::ProcessObject::DataObjectIdentifierType;

  /** Run-time type information (and related methods) */
  itkTypeMacro( CompositeImageNanMaskingFilter, InPlaceImageFilter )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      InputImageType::ImageDimension );

  itkNewMacro(Self)

  /** Set the compositve image image */
  itkNamedInputMacro(InputCompositeImage, InputImageType, "Primary")

  /** Set the moving image */
  itkNamedInputMacro(InputMaskImage, MaskImageType, "mask")

  /** Get the composite image output - this is the main output */
  itkNamedOutputMacro(OutputCompositeImage, InputImageType, "Primary")

  /** Get the mask output. */
  itkNamedOutputMacro(OutputMaskImage, MaskImageType, "mask")

  /** Since this filter has multiple outputs, it must reimplement MakeOutput() */
  DataObjectPointer MakeOutput(const DataObjectIdentifierType &) override;

  /** Graft outputs onto inputs */
  void AllocateOutputs() override;

  /** Main worker method */
  void DynamicThreadedGenerateData(const RegionType& outputRegionForThread) override;

protected:

  CompositeImageNanMaskingFilter();
  ~CompositeImageNanMaskingFilter() {}

private:
  CompositeImageNanMaskingFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};


#ifndef ITK_MANUAL_INSTANTIATION
#include "CompositeImageNanMaskingFilter.txx"
#endif


#endif // COMPOSITEIMAGENANMASKINGFILTER_H
