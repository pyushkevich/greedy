#ifndef ONEDIMENSIONALINPLACEACCUMULATEFILTER_H
#define ONEDIMENSIONALINPLACEACCUMULATEFILTER_H

#include "itkInPlaceImageFilter.h"
#include "itkImageRegionSplitterDirection.h"

/**
 * This is a filter for fast computation of box sums in an image. It is mean to be
 * used once in each image dimension (i.e., a separable filter). The input to the
 * filter is assumed to be a VectorImage (the filter is optimized for this)
 */
template <class TInputImage>
class OneDimensionalInPlaceAccumulateFilter : public itk::InPlaceImageFilter<TInputImage, TInputImage>
{
public:

  typedef OneDimensionalInPlaceAccumulateFilter<TInputImage> Self;
  typedef itk::InPlaceImageFilter<TInputImage, TInputImage> Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  itkTypeMacro(OneDimensionalInPlaceAccumulateFilter, itk::InPlaceImageFilter)

  itkNewMacro(Self)

  /** Some convenient typedefs. */
  typedef TInputImage                                  InputImageType;
  typedef TInputImage                                  OutputImageType;
  typedef typename OutputImageType::Pointer            OutputImagePointer;
  typedef typename OutputImageType::RegionType         OutputImageRegionType;
  typedef typename OutputImageType::PixelType          OutputImagePixelType;
  typedef typename OutputImageType::InternalPixelType  OutputImageComponentType;

  /** We use a custom splitter */
  typedef itk::ImageRegionSplitterDirection    SplitterType;

  /** ImageDimension constant */
  itkStaticConstMacro(OutputImageDimension, unsigned int, TInputImage::ImageDimension);

  itkGetMacro(Radius, int)
  itkSetMacro(Radius, int)

  itkGetMacro(Dimension, int)
  itkSetMacro(Dimension, int)

protected:

  OneDimensionalInPlaceAccumulateFilter();
  ~OneDimensionalInPlaceAccumulateFilter() {}

  virtual void ThreadedGenerateData(
      const OutputImageRegionType & outputRegionForThread,
      itk::ThreadIdType threadId);

  virtual const itk::ImageRegionSplitterBase *GetImageRegionSplitter(void) const;

  // Dimension of accumulation
  int m_Dimension;

  // Radius of accumulation
  int m_Radius;

  // Region splitter
  typename SplitterType::Pointer m_Splitter;

};



#ifndef ITK_MANUAL_INSTANTIATION
#include "OneDimensionalInPlaceAccumulateFilter.txx"
#endif


#endif // ONEDIMENSIONALINPLACEACCUMULATEFILTER_H
