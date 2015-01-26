#ifndef __ImageRegionConstIteratorWithIndexOverride_h_
#define __ImageRegionConstIteratorWithIndexOverride_h_

#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageLinearIteratorWithIndex.h"

template <class TIterator>
class IteratorExtender : public TIterator
{
public:
  typedef IteratorExtender<TIterator> Self;
  typedef TIterator Superclass;
  typedef typename Superclass::RegionType RegionType;
  typedef typename Superclass::ImageType ImageType;
  typedef typename Superclass::InternalPixelType InternalPixelType;


  IteratorExtender(ImageType *image, const RegionType &region)
    : Superclass(image, region) {}

  const InternalPixelType *GetPosition() { return this->m_Position; }

  const InternalPixelType *GetBeginPosition() { return this->m_Begin; }

};

template <class TIterator>
class IteratorExtenderWithOffset : public IteratorExtender<TIterator>
{
public:
  typedef IteratorExtender<TIterator> Superclass;

  typedef typename Superclass::RegionType RegionType;
  typedef typename Superclass::ImageType ImageType;
  typedef typename TIterator::OffsetValueType OffsetValueType;

  IteratorExtenderWithOffset(ImageType *image, const RegionType &region)
    : Superclass(image, region) {}

  const OffsetValueType GetOffset(int direction) { return this->m_OffsetTable[direction]; }
};


#endif
