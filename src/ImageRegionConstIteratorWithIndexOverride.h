#ifndef __ImageRegionConstIteratorWithIndexOverride_h_
#define __ImageRegionConstIteratorWithIndexOverride_h_

#include "itkImageRegionConstIteratorWithIndex.h"

template <typename TImage>
class ImageRegionConstIteratorWithIndexOverride
: public itk::ImageRegionConstIteratorWithIndex<TImage>
{
public:
  typedef ImageRegionConstIteratorWithIndexOverride<TImage> Self;
  typedef itk::ImageRegionConstIteratorWithIndex<TImage> Superclass;
  typedef typename Superclass::RegionType RegionType;
  typedef typename Superclass::InternalPixelType InternalPixelType;

  ImageRegionConstIteratorWithIndexOverride(TImage *im, const RegionType &region)
    : Superclass(im, region) {}

  const InternalPixelType *GetPosition() { return this->m_Position; }
  const InternalPixelType *GetBeginPosition() { return this->m_Begin; }
};


#endif
