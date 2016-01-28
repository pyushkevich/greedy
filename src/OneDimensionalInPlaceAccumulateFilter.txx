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
#include "OneDimensionalInPlaceAccumulateFilter.h"
#include <itkImageLinearIteratorWithIndex.h>
#include "ImageRegionConstIteratorWithIndexOverride.h"


template <class TInputImage>
OneDimensionalInPlaceAccumulateFilter<TInputImage>
::OneDimensionalInPlaceAccumulateFilter()
{
  m_Radius = 0;
  m_Dimension = 0;
  m_ComponentOffsetFront = m_ComponentOffsetBack = 0;
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


template <class TInputImage>
void
OneDimensionalInPlaceAccumulateFilter<TInputImage>
::SetComponentRange(int num_ignored_at_start, int num_ignored_at_end)
{
  m_ComponentOffsetFront = num_ignored_at_start;
  m_ComponentOffsetBack = num_ignored_at_end;
  this->Modified();
}

template <class T>
T* allocate_aligned(unsigned int n_elts)
{
  void *pointer;
  posix_memalign(&pointer, 16, sizeof(T) * n_elts);
  return static_cast<T*>(pointer);
}

#define _NCC_SSE_

#ifdef _NCC_SSE_

#include <xmmintrin.h>

template <class TPixel>
void
line_accumulate(
    TPixel *p_scan_pixel,
    TPixel *scanline,
    TPixel *p_scanline_end,
    TPixel *tailline,
    TPixel *sum_align,
    int line_length,
    int radius,
    int nc_used,
    int nc_padded,
    int kernel_width,
    int bytes_per_pixel,
    int padded_bytes_per_pixel,
    int jump)
{

}

template<>
void
line_accumulate<float>(
    float *p_scan_pixel,
    float *scanline,
    float *p_scanline_end,
    float *tailline,
    float *sum_align,
    int line_length,
    int radius,
    int nc_used,
    int nc_padded,
    int kernel_width,
    int bytes_per_pixel,
    int padded_bytes_per_pixel,
    int jump)
{
  typedef float OutputImageComponentType;
  int i, k;

  // Registers
  __m128 m_line, m_tail, m_sum_cur, m_sum_new;

  // Copy the contents of the image into the aligned line
  OutputImageComponentType *p_copy = scanline;
  const OutputImageComponentType *p_src = p_scan_pixel;
  for(; p_copy < p_scanline_end; p_copy += nc_padded, p_src += jump)
    {
    __builtin_prefetch(p_src + 5 * jump, 0, 0);
    for(i = 0; i < nc_used; i++)
      p_copy[i] = p_src[i];

    // memcpy(p_copy, p_src, bytes_per_pixel);
    }

  // Make a copy of the scan line
  for(p_src = scanline, p_copy = tailline; p_src < p_scanline_end; p_copy+=4, p_src+=4)
    {
    m_line = _mm_load_ps(p_src);
    _mm_store_ps(p_copy, m_line);
    }

  // Clear the sum array at the beginning
  for(int k = 0; k < nc_padded; k++)
    sum_align[k] = 0.0;

  // Pointer to the current position in the line
  OutputImageComponentType *p_line = scanline, *p_tail = tailline;

  // Pointer used for writing, it will trail the scan pointer
  OutputImageComponentType *p_write_pixel = scanline;

  // Pointer used for writing, it will trail the scan pointer
  OutputImageComponentType *p_sum_end = sum_align + nc_padded, *p_sum;

  // Compute the initial sum
  for(i = 0; i < radius; i++)
    {
    #pragma unroll
    for(p_sum = sum_align; p_sum < p_sum_end; p_sum+=4, p_line+=4)
      {
      m_line = _mm_load_ps(p_line);
      m_sum_cur = _mm_load_ps(p_sum);
      m_sum_new = _mm_add_ps(m_sum_cur, m_line);
      _mm_store_ps(p_sum, m_sum_new);
      }
    }

  // For the next Radius + 1 values, add to the sum and write
  for(; i < kernel_width; i++)
    {
    #pragma unroll
    for(p_sum = sum_align; p_sum < p_sum_end; p_sum+=4, p_line+=4, p_write_pixel+=4)
      {
      m_line = _mm_load_ps(p_line);
      m_sum_cur = _mm_load_ps(p_sum);
      m_sum_new = _mm_add_ps(m_sum_cur, m_line);
      _mm_store_ps(p_sum, m_sum_new);
      _mm_store_ps(p_write_pixel, m_sum_new);
      }
    }

  // Continue until we hit the end of the scanline
  for(; i < line_length; i++)
    {
    #pragma unroll
    for(p_sum = sum_align; p_sum < p_sum_end; p_sum+=4, p_line+=4, p_tail+=4, p_write_pixel+=4)
      {
      m_line = _mm_load_ps(p_line);
      m_tail = _mm_load_ps(p_tail);
      m_sum_cur = _mm_load_ps(p_sum);
      m_sum_new = _mm_add_ps(m_sum_cur, _mm_sub_ps(m_line, m_tail));
      _mm_store_ps(p_sum, m_sum_new);
      _mm_store_ps(p_write_pixel, m_sum_new);
      }
    }

  // Fill out the last bit
  for(; i < line_length + radius; i++)
    {
    #pragma unroll
    for(p_sum = sum_align; p_sum < p_sum_end; p_sum+=4, p_tail+=4, p_write_pixel+=4)
      {
      m_tail = _mm_load_ps(p_tail);
      m_sum_cur = _mm_load_ps(p_sum);
      m_sum_new = _mm_sub_ps(m_sum_cur, m_tail);
      _mm_store_ps(p_sum, m_sum_new);
      _mm_store_ps(p_write_pixel, m_sum_new);
      }
    }

  // Copy the accumulated pixels back into the main image
  OutputImageComponentType *p_copy_back = const_cast<OutputImageComponentType *>(p_scan_pixel);
  const OutputImageComponentType *p_src_back = scanline;
  for(; p_src_back < p_scanline_end; p_src_back += nc_padded, p_copy_back += jump)
    {
    __builtin_prefetch(p_copy_back + 5 * jump, 1, 0);
    for(i = 0; i < nc_used; i++)
      p_copy_back[i] = p_src_back[i];

    //
    // for(i = 0; i < nc; i++)
    //  p_copy_back[i] = p_src_back[i];

    // memcpy(p_copy_back, p_src_back, bytes_per_pixel);
    }
}

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

  // Get the first and last component for accumulation - these are optionally
  // specified by the user. The remaining components are left untouched
  int c_first = m_ComponentOffsetFront, c_last = (nc - 1) - m_ComponentOffsetBack;
  int n_skipped = m_ComponentOffsetFront + m_ComponentOffsetBack;

  // Get the offset corresponding to a move along the line for this iterator
  typename IteratorType::OffsetValueType jump = itLine.GetOffset(m_Dimension) * nc;

  // Length of the line being traversed (in whole pixels, then in components)
  int line_length = outputRegionForThread.GetSize(m_Dimension);
  int line_length_comp = line_length * nc;

  // Width of the kernel (in whole pixels, then in components)
  int kernel_width = 2 * m_Radius + 1;

  // We want some alignment for SIMD purposes. So we need to make a stride be a factor of 16 bytes
  int nc_used = nc - n_skipped;
  int bytes_per_pixel = sizeof(OutputImageComponentType) * (nc_used - n_skipped);

  // Round up, so it works out to 16 bytes
  int align_stride = 4 * sizeof(OutputImageComponentType);
  int padded_bytes_per_pixel = (bytes_per_pixel % align_stride) == 0
      ? bytes_per_pixel : align_stride * (1 + bytes_per_pixel / align_stride);

  // Number of chunks of four components per pixel
  int nc_padded = padded_bytes_per_pixel / sizeof(OutputImageComponentType);

  // This is a byte-aligned copy of the pixel column from the image
  OutputImageComponentType *scanline = allocate_aligned<OutputImageComponentType>(line_length * nc_padded);

  // This is a second aligned copy
  OutputImageComponentType *tailline = allocate_aligned<OutputImageComponentType>(line_length * nc_padded);

  // End of the scanline
  OutputImageComponentType *p_scanline_end = scanline + line_length * nc_padded;

  // Aligned sum array - where the sums are computed
  OutputImageComponentType *sum_align = allocate_aligned<OutputImageComponentType>(nc_padded);

  // Start iterating over lines
  for(itLine.GoToBegin(); !itLine.IsAtEnd(); itLine.NextLine())
    {
    int i, k;

    // Pointer to the beginning of the scan line
    long offset_in_pixels = itLine.GetPosition() - image->GetBufferPointer();
    long offset_in_comp = offset_in_pixels * nc;

    // Get the pointer to first component in first pixel
    const OutputImageComponentType *p_scan_pixel = image->GetBufferPointer() + offset_in_comp + c_first;

    // Run the main method
    line_accumulate<OutputImageComponentType>(
          const_cast<OutputImageComponentType *>(p_scan_pixel),
          scanline, p_scanline_end, tailline, sum_align,
          line_length, m_Radius, nc_used, nc_padded, kernel_width, bytes_per_pixel, padded_bytes_per_pixel, jump);

    }

  // Free allocated memory
  free(tailline);
  free(scanline);
  free(sum_align);
}


#else

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

  // Get the first and last component for accumulation - these are optionally
  // specified by the user
  int c_first = m_ComponentOffsetFront, c_last = (nc - 1) - m_ComponentOffsetBack;
  int n_skipped = m_ComponentOffsetFront + m_ComponentOffsetBack;

  // Get the offset corresponding to a move along the line for this iterator
  typename IteratorType::OffsetValueType jump = itLine.GetOffset(m_Dimension) * nc;

  // Length of the line being traversed (in whole pixels, then in components)
  int line_length = outputRegionForThread.GetSize(m_Dimension),
      line_length_comp = line_length * nc;

  // Width of the kernel (in whole pixels, then in components)
  int kernel_width = 2 * m_Radius + 1;

  // Allocate an array of the length of the line in components
  OutputImageComponentType *line = new OutputImageComponentType[line_length_comp];
  // double *line = new double[line_length_comp];
  // double *sum = new double[nc], *sum_end = sum + nc, *p_sum;

  // Allocate an array to hold the current running sum
  // OutputImageComponentType *sum = new OutputImageComponentType[nc], *sum_end = sum + nc, *p_sum;
  OutputImageComponentType *sum = new OutputImageComponentType[nc];

  // Pointers into the sum array for the included components
  OutputImageComponentType *sum_start = sum + c_first, *sum_end = sum + c_last + 1, *p_sum;

  // Two versions of the code - I thought that maybe the second version (further down) would be
  // more optimized by the compiler, but if anything, I see an opposite effect (although tiny)

#ifdef _ACCUM_ITER_CODE_

  // Start iterating over lines
  for(itLine.GoToBegin(); !itLine.IsAtEnd(); itLine.NextLine())
    {
    int i;

    // Initialize the sum to zero
    for(p_sum = sum_start; p_sum < sum_end; p_sum++)
      *p_sum  = itk::NumericTraits<OutputImageComponentType>::Zero;

    // Pointer to the current position in the line
    OutputImageComponentType *p_line = line + c_first, *p_tail = p_line;

    // Pointer to the beginning of the scan line
    long offset_in_pixels = itLine.GetPosition() - image->GetBufferPointer();
    long offset_in_comp = offset_in_pixels * nc;
    const OutputImageComponentType *p_scan_pixel = image->GetBufferPointer() + offset_in_comp + c_first, *p_scan;

    // Pointer used for writing, it will trail the scan pointer
    OutputImageComponentType *p_write_pixel = const_cast<OutputImageComponentType *>(p_scan_pixel), *p_write;

    // Compute the initial sum
    for(i = 0; i < m_Radius; i++)
      {
      for(p_scan = p_scan_pixel, p_sum = sum_start;
          p_sum < sum_end;
          p_sum++, p_line++, p_scan++)
        {
        *p_sum += *p_line = *p_scan;
        }

      p_scan_pixel += jump;
      p_line += n_skipped;
      }

    // For the next Radius + 1 values, add to the sum and write
    for(; i < kernel_width; i++)
      {
      for(p_scan = p_scan_pixel, p_write = p_write_pixel, p_sum = sum_start;
          p_sum < sum_end;
          p_sum++, p_line++, p_scan++, p_write++)
        {
        *p_line = *p_scan;
        *p_sum += *p_line;
        *p_write = *p_sum;
        }

      p_scan_pixel += jump;
      p_write_pixel += jump;
      p_line += n_skipped;
      }

    // Continue until we hit the end of the scanline
    for(; i < line_length; i++)
      {
      for(p_scan = p_scan_pixel, p_write = p_write_pixel, p_sum = sum_start;
          p_sum < sum_end;
          p_sum++, p_line++, p_scan++, p_write++, p_tail++)
        {
        *p_line = *p_scan;
        *p_sum += *p_line - *p_tail;
        *p_write = *p_sum;
        }

      p_scan_pixel += jump;
      p_write_pixel += jump;
      p_line += n_skipped;
      p_tail += n_skipped;
      }

    // Fill out the last bit
    for(; i < line_length + m_Radius; i++)
      {
      for(p_write = p_write_pixel, p_sum = sum_start;
          p_sum < sum_end;
          p_sum++, p_write++, p_tail++)
        {
        *p_sum -= *p_tail;
        *p_write = *p_sum;
        }

      p_write_pixel += jump;
      p_tail += n_skipped;
      }
    }

#else

  // Start iterating over lines
  for(itLine.GoToBegin(); !itLine.IsAtEnd(); itLine.NextLine())
    {

    int i, k, m;

    // Initialize the sum to zero
    for(int k = c_first; k <= c_last; k++)
      sum[k] = itk::NumericTraits<OutputImageComponentType>::Zero;



    // Pointer to the current position in the line
    OutputImageComponentType *p_line = line, *p_tail = p_line;

    // Pointer to the beginning of the scan line
    long offset_in_pixels = itLine.GetPosition() - image->GetBufferPointer();
    long offset_in_comp = offset_in_pixels * nc;

    // Where we are scanning from
    const OutputImageComponentType *p_scan_pixel = image->GetBufferPointer() + offset_in_comp;

    // Pointer used for writing, it will trail the scan pointer
    OutputImageComponentType *p_write_pixel = const_cast<OutputImageComponentType *>(p_scan_pixel);

    // Compute the initial sum
    for(i = 0, m = 0; i < m_Radius; i++)
      {
      for(k = c_first; k <= c_last; k++)
        {
        sum[k] += p_line[k] = p_scan_pixel[k];
        }
      p_scan_pixel += jump;
      p_line += nc;
      }

    // For the next Radius + 1 values, add to the sum and write
    for(; i < kernel_width; i++)
      {
      for(k = c_first; k <= c_last; k++)
        {
        p_write_pixel[k] = (sum[k] += p_line[k] = p_scan_pixel[k]);
        }

      p_scan_pixel += jump;
      p_write_pixel += jump;
      p_line += nc;
      }

    // Continue until we hit the end of the scanline
    for(; i < line_length; i++)
      {
      for(k = c_first; k <= c_last; k++)
        {
        p_write_pixel[k] = (sum[k] += (p_line[k] = p_scan_pixel[k]) - p_tail[k]);
        }

      p_scan_pixel += jump;
      p_write_pixel += jump;
      p_line += nc;
      p_tail += nc;
      }

    // Fill out the last bit
    for(; i < line_length + m_Radius; i++)
      {
      for(k = c_first; k <= c_last; k++)
        {
        p_write_pixel[k] = (sum[k] -= p_tail[k]);
        }

      p_write_pixel += jump;
      p_tail += nc;
      }
    }
#endif

  delete sum;
  delete line;
}


#endif  // _NCC_SSE_


template <class TInputImage>
typename TInputImage::Pointer
AccumulateNeighborhoodSumsInPlace(TInputImage *image, const typename TInputImage::SizeType &radius,
                                  int num_ignored_at_start, int num_ignored_at_end)
{
  typedef OneDimensionalInPlaceAccumulateFilter<TInputImage> AccumFilterType;

  typename itk::ImageSource<TInputImage>::Pointer pipeTail;
  for(int dir = 0; dir < TInputImage::ImageDimension; dir++)
    {
    typename AccumFilterType::Pointer accum = AccumFilterType::New();
    accum->SetInput(pipeTail.IsNull() ? image : pipeTail->GetOutput());
    accum->SetDimension(dir);
    accum->SetRadius(radius[dir]);
    accum->SetComponentRange(num_ignored_at_start, num_ignored_at_end);
    pipeTail = accum;

    accum->Update();
    }

  return pipeTail->GetOutput();
}
