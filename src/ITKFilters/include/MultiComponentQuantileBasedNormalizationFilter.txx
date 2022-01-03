#ifndef MULTICOMPONENTQUANTILEBASEDNORMALIZATIONFILTER_TXX
#define MULTICOMPONENTQUANTILEBASEDNORMALIZATIONFILTER_TXX

#include "MultiComponentQuantileBasedNormalizationFilter.h"
#include "ImageRegionConstIteratorWithIndexOverride.h"

template <class TInputImage, class TOutputImage>
MultiComponentQuantileBasedNormalizationFilter<TInputImage, TOutputImage>
::MultiComponentQuantileBasedNormalizationFilter()
{
}

template <class TInputImage, class TOutputImage>
void
MultiComponentQuantileBasedNormalizationFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();
  this->GetOutput()->SetNumberOfComponentsPerPixel(this->GetInput()->GetNumberOfComponentsPerPixel());
}

template <class TInputImage, class TOutputImage>
void
MultiComponentQuantileBasedNormalizationFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  unsigned int ncomp = this->GetInput()->GetNumberOfComponentsPerPixel();
  m_LowerQuantileValues.resize(ncomp);
  m_UpperQuantileValues.resize(ncomp);
  m_NumberOfNaNs.resize(ncomp);

  // Adjust the out of range values
  if(!m_UseLowerOutOfRangeOutputValue)
    m_LowerOutOfRangeOutputValue = m_LowerQuantileOutputValue;
  if(!m_UseUpperOutOfRangeOutputValue)
    m_UpperOutOfRangeOutputValue = m_UpperQuantileOutputValue;
}

template <class TInputImage, class TOutputImage>
void
MultiComponentQuantileBasedNormalizationFilter<TInputImage, TOutputImage>
::GenerateData()
{
  // Standard stuff done before splitting into threads
  this->AllocateOutputs();
  this->BeforeThreadedGenerateData();

  // Iterator typdef
  typedef itk::ImageLinearConstIteratorWithIndex<InputImageType> IterBase;
  typedef IteratorExtender<IterBase> Iterator;

  // Determine the size of the heap
  long total_pixels = this->GetInput()->GetBufferedRegion().GetNumberOfPixels();
  long heap_size_upper = 1 + (int)((1.0 - m_UpperQuantile) * total_pixels);
  long heap_size_lower = 1 + (int)(m_LowerQuantile * total_pixels);

  int ncomp = this->GetInput()->GetNumberOfComponentsPerPixel();

  // Mutex for combining heaps
  std::mutex heap_mutex;

  // Repeat for each component
  for(int k = 0; k < ncomp; k++)
    {
    // Global heaps for the whole image for this component
    ThreadData td;

    // Use parallelization to compute the upper and lower heaps for each parallel region
    itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
    mt->ParallelizeImageRegion<Self::ImageDimension>(
          this->GetOutput()->GetBufferedRegion(),
          [this,k,&td,heap_size_lower,heap_size_upper,ncomp,&heap_mutex](const OutputImageRegionType &region)
      {
      // Heaps for this region
      ThreadData td_local;
      long line_length = region.GetSize(0);

      // Iterate over the image to update the heaps
      for(Iterator it(this->GetInput(), region); !it.IsAtEnd(); it.NextLine())
        {
        // Get a pointer to the start of the line
        const InputComponentType *line = it.GetPixelPointer(this->GetInput()) + k;

        // Iterate over the line
        for(int p = 0; p < line_length; p++, line+=ncomp)
          {
          InputComponentType v = *line;
          if(!std::isnan(v))
            {
            heap_lower_push(td_local.heap_lower, heap_size_lower, v);
            heap_upper_push(td_local.heap_upper, heap_size_upper, v);
            }
          else
            {
            td_local.number_of_nans++;
            }
          }
        }

      // Use mutex to update the global heaps
      std::lock_guard<std::mutex> guard(heap_mutex);

      // We can now write safely to td
      while(!td_local.heap_lower.empty())
        {
        InputComponentType v = td_local.heap_lower.top();
        heap_lower_push(td.heap_lower, heap_size_lower, v);
        td_local.heap_lower.pop();
        }

      while(!td_local.heap_upper.empty())
        {
        InputComponentType v = td_local.heap_upper.top();
        heap_upper_push(td.heap_upper, heap_size_upper, v);
        td_local.heap_upper.pop();
        }

      td.number_of_nans += td_local.number_of_nans;
    }, nullptr);

    // At this point, a combined heap has been generated

    // Update the heap size based on the number of nans
    long nonnan_pixels = total_pixels - td.number_of_nans;
    long heap_size_upper_upd = 1 + (int)((1.0 - m_UpperQuantile) * nonnan_pixels);
    long heap_size_lower_upd = 1 + (int)(m_LowerQuantile * nonnan_pixels);

    // Pop until the heap is the right size
    while(td.heap_upper.size() > heap_size_upper_upd)
      td.heap_upper.pop();

    while(td.heap_lower.size() > heap_size_lower_upd)
      td.heap_lower.pop();

    // Get the quantile values
    m_UpperQuantileValues[k] = td.heap_upper.top();
    m_LowerQuantileValues[k] = td.heap_lower.top();
    m_NumberOfNaNs[k] = td.number_of_nans;

    // Continue if no remapping requested
    if(m_NoRemapping)
      continue;

    // Compute the scale and shift
    double scale = (m_UpperQuantileOutputValue - m_LowerQuantileOutputValue) * 1.0
                   / (m_UpperQuantileValues[k] - m_LowerQuantileValues[k]);

    double shift = m_LowerQuantileValues[k] * scale - m_LowerQuantileOutputValue;

    // Perform remapping on the image in parallel
    mt->ParallelizeImageRegion<Self::ImageDimension>(
          this->GetOutput()->GetBufferedRegion(),
          [this,k,scale,shift,ncomp](const OutputImageRegionType &region)
      {
      // Now each thread remaps the intensities into the quantile range
      for(Iterator it(this->GetInput(), region); !it.IsAtEnd(); it.NextLine())
        {
        // Get a pointer to the start of the line
        const InputComponentType *line = it.GetPixelPointer(this->GetInput()) + k;
        OutputComponentType *out_line = it.GetPixelPointer(this->GetOutput()) + k;

        // Iterate over the line
        long line_length = region.GetSize(0);
        for(int p = 0; p < line_length; p++, line+=ncomp, out_line+=ncomp)
          {
          OutputComponentType value = (OutputComponentType) (*line * scale - shift);
          if(value < m_LowerQuantileOutputValue)
            *out_line = m_LowerOutOfRangeOutputValue;
          else if(value > m_UpperQuantileOutputValue)
            *out_line = m_UpperOutOfRangeOutputValue;
          else
            *out_line = value;
          }
        }
      }, nullptr);
    } // loop over components

  this->AfterThreadedGenerateData();
}




#endif // MULTICOMPONENTQUANTILEBASEDNORMALIZATIONFILTER_TXX
