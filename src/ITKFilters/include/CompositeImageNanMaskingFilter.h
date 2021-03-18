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
  typename itk::DataObject::Pointer MakeOutput(const typename Superclass::DataObjectIdentifierType &) override;

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
