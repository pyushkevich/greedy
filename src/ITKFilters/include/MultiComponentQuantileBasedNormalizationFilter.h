#ifndef MULTICOMPONENTQUANTILEBASEDNORMALIZATIONFILTER_H
#define MULTICOMPONENTQUANTILEBASEDNORMALIZATIONFILTER_H

#include <itkImageToImageFilter.h>
#include <queue>

/**
 * A helper filter to remap intensities for mutual information. It can double
 * as a quick way to compute per-component quantiles of a multi-component image
 */
template <class TInputImage, class TOutputImage>
class MultiComponentQuantileBasedNormalizationFilter
    : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  typedef MultiComponentQuantileBasedNormalizationFilter<TInputImage, TOutputImage> Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiComponentQuantileBasedNormalizationFilter, ImageToImageFilter )

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename Superclass::InputImageType                InputImageType;
  typedef typename InputImageType::PixelType                 InputPixelType;
  typedef typename InputImageType::InternalPixelType         InputComponentType;
  typedef typename Superclass::OutputImageType               OutputImageType;
  typedef typename OutputImageType::PixelType                OutputPixelType;
  typedef typename OutputImageType::InternalPixelType        OutputComponentType;

  typedef typename InputImageType::IndexType                 IndexType;
  typedef typename InputImageType::SizeType                  SizeType;


  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, InputImageType::ImageDimension );

  /**
   * Set the lower quantile (default 0), below which all values are treated as equal
   * to the minimum value.
   */
  itkSetMacro(LowerQuantile, double)

  /**
   * Set the upper quantile (default 0.99), above which all values are treated as equal
   * to the maximum value.
   */
  itkSetMacro(UpperQuantile, double)

  /**
   * Set the output value to which the lower quantile maps to
   */
  itkSetMacro(LowerQuantileOutputValue, OutputComponentType)

  /**
   * Set the output value to which the upper quantile maps to
   */
  itkSetMacro(UpperQuantileOutputValue, OutputComponentType)

  /**
   * Set the lower out of range value (by default this is the same as
   * the lower quantile output value)
   */
  void SetLowerOutOfRangeOutputValue(const OutputComponentType &val)
  {
    this->m_LowerOutOfRangeOutputValue = val;
    this->m_UseLowerOutOfRangeOutputValue = true;
    this->Modified();
  }

  /**
   * Set the upper out of range value (by default this is the same as
   * the upper quantile output value)
   */
  void SetUpperOutOfRangeOutputValue(const OutputComponentType &val)
  {
    this->m_UpperOutOfRangeOutputValue = val;
    this->m_UseUpperOutOfRangeOutputValue = true;
    this->Modified();
  }

  /**
   * When this flag is set, the quantiles are computed and nothing else is done, i.e., the
   * input image is passed on as is. Set the filter to be in place.
   */
  itkSetMacro(NoRemapping, bool)

  /**
   * When this flag is set, the lowest intensity in the image is mapped to bin 1, rather than
   * bin 0. This leaves bin 0 in the histogram empty. Subsequently, it can be used to represent
   * outside values. In other words, when this is set, the remapped image will have intensities
   * between 1 and n_Bins-1, when it is not set, this will be between 0 and n_Bins-1
   *
   * By default this flag is False.
   */
  // itkSetMacro(StartAtBinOne, bool)

  /** After the filter ran, get the value of the lower quantile */
  InputComponentType GetLowerQuantileValue(int component) const
    { return m_LowerQuantileValues[component]; }

  /** After the filter ran, get the value of the upper quantile */
  InputComponentType GetUpperQuantileValue(int component) const
    { return m_UpperQuantileValues[component]; }

  /** After the filter ran, get the number of NaN pixels */
  unsigned long GetNumberOfNaNs(int component) const
    { return m_NumberOfNaNs[component]; }

protected:
  MultiComponentQuantileBasedNormalizationFilter();
  ~MultiComponentQuantileBasedNormalizationFilter() {}

  virtual void GenerateOutputInformation() ITK_OVERRIDE;

  virtual void BeforeThreadedGenerateData() ITK_OVERRIDE;
  virtual void GenerateData() ITK_OVERRIDE;

private:
  MultiComponentQuantileBasedNormalizationFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // The quantile to map to the upper and lower bins.
  double m_LowerQuantile = 0.0, m_UpperQuantile = 0.99;

  // Output values
  OutputComponentType m_LowerQuantileOutputValue = 0, m_UpperQuantileOutputValue = 255;

  // Out of range output values
  OutputComponentType m_LowerOutOfRangeOutputValue = 0, m_UpperOutOfRangeOutputValue = 255;

  // Are these values used?
  bool m_UseLowerOutOfRangeOutputValue = false, m_UseUpperOutOfRangeOutputValue = false;

  // Heap data types
  typedef std::priority_queue<
    InputComponentType, std::vector<InputComponentType>, std::less<InputComponentType> > LowerHeap;

  typedef std::priority_queue<
    InputComponentType, std::vector<InputComponentType>, std::greater<InputComponentType> > UpperHeap;

  void heap_lower_push(LowerHeap &heap, int max_size, const InputComponentType &v)
  {
    if(heap.size() < max_size)
      {
      heap.push(v);
      }
    else if(heap.top() > v)
      {
      heap.pop();
      heap.push(v);
      }
  }

  void heap_upper_push(UpperHeap &heap, int max_size, const InputComponentType &v)
  {
    if(heap.size() < max_size)
      {
      heap.push(v);
      }
    else if(heap.top() < v)
      {
      heap.pop();
      heap.push(v);
      }
  }

  // Per thread data
  struct ThreadData
  {
    // Heaps for minimum and maximum intensities
    LowerHeap heap_lower;
    UpperHeap heap_upper;
    unsigned long number_of_nans = 0ul;
  };

  std::vector<ThreadData> m_ThreadData;

  std::vector<InputComponentType> m_LowerQuantileValues, m_UpperQuantileValues;
  std::vector<unsigned long> m_NumberOfNaNs;

  bool m_NoRemapping = false;

  // bool m_StartAtBinOne;
};

#ifndef ITK_MANUAL_INSTANTIATION
#include "MultiComponentQuantileBasedNormalizationFilter.txx"
#endif

#endif // MULTICOMPONENTQUANTILEBASEDNORMALIZATIONFILTER_H
