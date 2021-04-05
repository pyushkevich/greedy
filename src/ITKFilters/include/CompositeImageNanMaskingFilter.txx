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
#ifndef COMPOSITEIMAGENANMASKINGFILTER_TXX
#define COMPOSITEIMAGENANMASKINGFILTER_TXX

#include "CompositeImageNanMaskingFilter.h"
#include "ImageRegionConstIteratorWithIndexOverride.h"

template <class TCompositeImage, class TMaskImage>
CompositeImageNanMaskingFilter<TCompositeImage, TMaskImage>
::CompositeImageNanMaskingFilter()
{
  this->InPlaceOn();
  this->SetPrimaryOutput(this->MakeOutput("Primary"));
  this->SetOutput("mask", this->MakeOutput("mask"));
}

template <class TCompositeImage, class TMaskImage>
typename CompositeImageNanMaskingFilter<TCompositeImage, TMaskImage>::DataObjectPointer
CompositeImageNanMaskingFilter<TCompositeImage, TMaskImage>
::MakeOutput(const DataObjectIdentifierType &key)
{
  if(key == "Primary")
    return (InputImageType::New()).GetPointer();
  else if(key == "mask")
    return (MaskImageType::New()).GetPointer();
  else
    return nullptr;
}


template <class TCompositeImage, class TMaskImage>
void
CompositeImageNanMaskingFilter<TCompositeImage, TMaskImage>
::AllocateOutputs()
{
  this->GetOutputCompositeImage()->Graft(this->GetInputCompositeImage());
  this->GetOutputMaskImage()->Graft(this->GetInputMaskImage());
}

template <class TCompositeImage, class TMaskImage>
void
CompositeImageNanMaskingFilter<TCompositeImage, TMaskImage>
::DynamicThreadedGenerateData(const RegionType& outputRegionForThread)
{
  // Get the input image and mask image
  InputImageType *image = this->GetOutputCompositeImage();
  MaskImageType *mask = this->GetOutputMaskImage();

  // Create an iterator that runs over lines in the input image
  typedef itk::ImageLinearIteratorWithIndex<InputImageType> IterBase;
  typedef IteratorExtender<IterBase> Iter;
  unsigned int nc = image->GetNumberOfComponentsPerPixel();
  unsigned int line_len = outputRegionForThread.GetSize()[0];
  for(Iter it(image, outputRegionForThread); !it.IsAtEnd(); it.NextLine())
    {
    // Get the pointer to the beginning of the pixel line
    typename InputImageType::InternalPixelType *p = it.GetPixelPointer(image);
    typename InputImageType::InternalPixelType *p_end = p + nc * line_len;
    typename InputImageType::InternalPixelType *p_mask = it.GetPixelPointer(mask);

    // Go over pixels in this line
    for(; p < p_end; p+=nc, p_mask++)
      {
      // If mask is already zero, just zero out the intensities
      bool clear_intensities = false;
      if(*p_mask == 0.0)
        {
        clear_intensities = true;
        }
      else
        {
        for(unsigned int j = 0; j < nc; j++)
          {
          if(std::isnan(p[j]))
            {
            clear_intensities = true;
            *p_mask = 0.0;
            break;
            }
          }
        }

      // Clear intensities if needed
      if(clear_intensities)
        for(unsigned int j = 0; j < nc; j++)
          p[j] = 0;
      }
    }
}

#endif // COMPOSITEIMAGENANMASKINGFILTER_TXX
