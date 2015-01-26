#include "OneDimensionalInPlaceAccumulateFilter.h"
#include <itkImageLinearIteratorWithIndex.h>


template <class TInputImage>
OneDimensionalInPlaceAccumulateFilter<TInputImage>
::OneDimensionalInPlaceAccumulateFilter()
{
  m_Radius = 0;
  m_Dimension = 0;
  m_Splitter = SplitterType::New();
  this->InPlaceOn();
}

template <class TInputImage>
const itk::ImageRegionSplitterBase *
OneDimensionalInPlaceAccumulateFilter<TInputImage>
::GetImageRegionSplitter(void) const
{
  m_Splitter->SetDirection(m_Dimension);
  return m_Splitter;
}

#include "ImageRegionConstIteratorWithIndexOverride.h"

template <class TInputImage>
void
OneDimensionalInPlaceAccumulateFilter<TInputImage>
::ThreadedGenerateData(
    const OutputImageRegionType & outputRegionForThread,
    itk::ThreadIdType threadId)
{
  // Get the image
  InputImageType *image = const_cast<InputImageType *>(this->GetInput());

  // Set up the iterator that will go through all the lines in the
  // output region. We assume that the lines span the whole length of
  // the input, i.e., the threading direction does not interfere
  typedef itk::ImageLinearIteratorWithIndex<TInputImage> IteratorBaseType;
  typedef IteratorExtenderWithOffset<IteratorBaseType> IteratorType;

  // This is the line iterator, although for even greater speed we operate
  // directly on pointers, so we only use it's NextLine functionality()
  IteratorType itLine(image, outputRegionForThread);
  itLine.SetDirection(m_Dimension);

  // Get the number of components
  int nc = image->GetNumberOfComponentsPerPixel();

  // Get the offset corresponding to a move along the line for this iterator
  typename IteratorType::OffsetValueType jump = itLine.GetOffset(m_Dimension) * nc;

  // Length of the line being traversed (in whole pixels, then in components)
  int line_length = outputRegionForThread.GetSize(m_Dimension),
      line_length_comp = line_length * nc;

  // Width of the kernel (in whole pixels, then in components)
  int kernel_width = 2 * m_Radius + 1;

  // Allocate an array of the length of the line in components
  OutputImageComponentType *line = new OutputImageComponentType[line_length_comp];

  // Allocate an array to hold the current running sum
  OutputImageComponentType *sum = new OutputImageComponentType[nc], *sum_end = sum + nc, *p_sum;

  // Start iterating over lines
  for(itLine.GoToBegin(); !itLine.IsAtEnd(); itLine.NextLine())
    {
    int i;

    // Initialize the sum to zero
    for(p_sum = sum; p_sum < sum_end; p_sum++)
      *p_sum  = itk::NumericTraits<OutputImageComponentType>::Zero;

    // Pointer to the current position in the line
    OutputImageComponentType *p_line = line, *p_tail = line;

    // Pointer to the beginning of the scan line
    long offset_in_pixels = itLine.GetPosition() - itLine.GetBeginPosition();
    long offset_in_comp = offset_in_pixels * nc;
    const OutputImageComponentType *p_scan_pixel = image->GetBufferPointer() + offset_in_comp, *p_scan;

    // Pointer used for writing, it will trail the scan pointer
    OutputImageComponentType *p_write_pixel = const_cast<OutputImageComponentType *>(p_scan_pixel), *p_write;

    // Compute the initial sum
    for(i = 0; i < m_Radius; i++)
      {
      for(p_scan = p_scan_pixel, p_sum = sum;
          p_sum < sum_end;
          p_sum++, p_line++, p_scan++)
        {
        *p_sum += *p_line = *p_scan;
        }

      p_scan_pixel += jump;
      }

    // For the next Radius + 1 values, add to the sum and write
    for(; i < kernel_width; i++)
      {
      for(p_scan = p_scan_pixel, p_write = p_write_pixel, p_sum = sum;
          p_sum < sum_end;
          p_sum++, p_line++, p_scan++, p_write++)
        {
        *p_line = *p_scan;
        *p_sum += *p_line;
        *p_write = *p_sum;
        }

      p_scan_pixel += jump;
      p_write_pixel += jump;
      }

    // Continue until we hit the end of the scanline
    for(; i < line_length; i++)
      {
      for(p_scan = p_scan_pixel, p_write = p_write_pixel, p_sum = sum;
          p_sum < sum_end;
          p_sum++, p_line++, p_scan++, p_write++, p_tail++)
        {
        *p_line = *p_scan;
        *p_sum += *p_line - *p_tail;
        *p_write = *p_sum;
        }

      p_scan_pixel += jump;
      p_write_pixel += jump;
      }

    // Fill out the last bit
    for(; i < line_length + m_Radius; i++)
      {
      for(p_write = p_write_pixel, p_sum = sum;
          p_sum < sum_end;
          p_sum++, p_write++, p_tail++)
        {
        *p_sum -= *p_tail;
        *p_write = *p_sum;
        }

      p_write_pixel += jump;
      }
    }

  delete sum;
  delete line;
}

