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
#ifndef MULTICOMPONENTMUTUALINFOIMAGEMETRIC_TXX
#define MULTICOMPONENTMUTUALINFOIMAGEMETRIC_TXX

#include "MultiComponentMutualInfoImageMetric.h"

template <class TMetricTraits>
void
MultiComponentMutualInfoImageMetric<TMetricTraits>
::BeforeThreadedGenerateData()
{
  Superclass::BeforeThreadedGenerateData();

  int ncomp = this->GetFixedImage()->GetNumberOfComponentsPerPixel();

  // Clear the per-thread histograms
  m_MIThreadData.clear();

  // Create the per-thread histograms
  for(int k = 0; k < this->GetNumberOfThreads(); k++)
    {
    m_MIThreadData.push_back(MutualInfoThreadData());
    m_MIThreadData.back().m_Histogram.Initialize(ncomp, m_Bins);
    }

  // Initialize the weight storage
  if(this->m_ComputeGradient)
    m_GradWeights.Initialize(ncomp, m_Bins);


  // Code to determine the actual number of threads used below
  itk::ThreadIdType nbOfThreads = this->GetNumberOfThreads();
  if ( itk::MultiThreader::GetGlobalMaximumNumberOfThreads() != 0 )
    {
    nbOfThreads = vnl_math_min( this->GetNumberOfThreads(), itk::MultiThreader::GetGlobalMaximumNumberOfThreads() );
    }

  itk::ImageRegion<ImageDimension> splitRegion;  // dummy region - just to call
                                                  // the following method
  nbOfThreads = this->SplitRequestedRegion(0, nbOfThreads, splitRegion);

  // Initialize the barrier
  m_Barrier = itk::Barrier::New();
  m_Barrier->Initialize(nbOfThreads);
}

template <class TMetricTraits>
void
MultiComponentMutualInfoImageMetric<TMetricTraits>
::AfterThreadedGenerateData()
{
  Superclass::AfterThreadedGenerateData();
  for(int k = 0; k < this->GetNumberOfThreads(); k++)
    {
    m_MIThreadData.back().m_Histogram.Deallocate();
    m_MIThreadData.pop_back();
    }
}


template <class TMetricTraits>
void
MultiComponentMutualInfoImageMetric<TMetricTraits>
::ThreadedGenerateData(
    const OutputImageRegionType &outputRegionForThread,
    itk::ThreadIdType threadId)
{
  // Get the number of components
  int ncomp = this->GetFixedImage()->GetNumberOfComponentsPerPixel();

  // Create an iterator specialized for going through metrics
  typedef MultiComponentMetricWorker<TMetricTraits, MetricImageType> InterpType;
  InterpType iter(this, this->GetMetricOutput(), outputRegionForThread);

  // Initially, I am implementing this as a two-pass filter. On the first pass, the joint
  // histogram is computed without the gradient. On the second pass, the gradient is computed.
  // The inefficiency of this implementation is that the interpolation code is being called
  // twice. The only way I see to avoid this is to store the results of each interpolation in
  // an intermediate working image, but I am not sure how much one would save from that!

  // First pass - compute the histograms
  MutualInfoThreadData &td = m_MIThreadData[threadId];

  // Iterate over the lines
  for(; !iter.IsAtEnd(); iter.NextLine())
    {
    // Iterate over the pixels in the line
    for(; !iter.IsAtEndOfLine(); ++iter)
      {
      // Get the current histogram corners
      if(iter.CheckFixedMask())
        iter.PartialVolumeHistogramSample(td.m_Histogram);
      }
    }

  // Wait for all the threads to finish this computation
  m_Barrier->Wait();

  // Add up and process the histograms - use the first one as the target for storage
  if(threadId == 0)
    {
    // Initialize the histograms per component
    m_Histograms.clear();

    // All procesing is separate for each component
    for(int c = 0; c < ncomp; c++)
      {
      // The histogram for this component
      m_Histograms.push_back(Histogram(m_Bins));
      Histogram &hc = m_Histograms.back();

      // The total sum of entries
      double hist_sum = this->GetOutput()->GetBufferedRegion().GetNumberOfPixels();

      // Process each entry
      for(int bf = 0; bf < m_Bins; bf++)
        {
        for(int bm = 0; bm < m_Bins; bm++)
          {
          // Reference to the joint probability entry
          double &Pfm = hc.Pfm(bf,bm);

          // Add the entries from all threads
          for(int q = 0; q < this->GetNumberOfThreads(); q++)
            Pfm += m_MIThreadData[q].m_Histogram[c][bf][bm];

          // Normalize to make a probability
          Pfm /= hist_sum;

          // Add up the marginals
          hc.Pf[bf] += Pfm;
          hc.Pm[bm] += Pfm;
          }
        }

      // Compute the mutual information for this component and overall
      double &m_comp = this->m_ThreadData[0].comp_metric[c];
      double &m_total = this->m_ThreadData[0].metric;

      for(int bf = 0; bf < m_Bins; bf++)
        {
        for(int bm = 0; bm < m_Bins; bm++)
          {
          double Pfm = hc.Pfm(bf, bm);

          if(Pfm > 0)
            {
            // TODO: temp change
            double q = log(Pfm / (hc.Pf(bf) * hc.Pm(bm)));
            double v = Pfm * q;
            m_comp += v;
            m_total += v;

            // If computing the gradient, also compute the additional weight information
            if(this->m_ComputeGradient)
              {
              // TODO: scale by weights!
              m_GradWeights[c][bf][bm] = log(Pfm / hc.Pm(bm));
              }
            }
          }
          /* // code with epsilon trick
          double Pfm = hc.Pfm(bf, bm);
          double Pm = hc.Pm(bm), Pf = hc.Pf(bf);
          double eps = 1.0e-8;

          double v = (Pfm + eps) * log(Pfm + eps) -
                     ((Pf + eps) * log(Pf + eps) + (Pm + eps) * log(Pm + eps)) / m_Bins;

          m_comp += v;
          m_total += v;

          if(this->m_ComputeGradient)
            {
            m_GradWeights[c][bf][bm] = log((Pfm + eps) / (Pm + eps));
            }
          } */
        }

          /*
          if(Pfm > 0)
            {
            // TODO: temp change
            double q = log(Pfm / (hc.Pf(bf) * hc.Pm(bm)));
            // double q = log(Pfm);
            double v = Pfm * q;
            m_comp += v;
            m_total += v;

            // If computing the gradient, also compute the additional weight information
            if(this->m_ComputeGradient)
              {
              // TODO: scale by weights!
              // TODO: temp change
              // m_GradWeights[c][bf][bm] = 1.0 + q - Pfm / hc.Pm(bm);
              // m_GradWeights[c][bf][bm] = 1.0 + q;
              m_GradWeights[c][bf][bm] = log(Pfm) - log(hc.Pm(bm));
              }
            }
          }
        }
            */
      }

    // The last thing is to set the normalizing constant to 1
    this->m_ThreadData[0].mask = 1.0;
    }

  // Wait for all threads
  m_Barrier->Wait();

  // At this point, we should be computing the gradient using the probability values computed above
  if(this->m_ComputeGradient && !this->m_ComputeAffine)
    {
    GradientPixelType *grad_buffer = this->GetDeformationGradientOutput()->GetBufferPointer();

    // Iterate one more time through the voxels
    InterpType iter_g(this, this->GetMetricOutput(), outputRegionForThread);
    for(; !iter_g.IsAtEnd(); iter_g.NextLine())
      {
      // Get the output gradient pointer at the beginning of this line
      GradientPixelType *grad_line = iter_g.GetOffsetInPixels() + grad_buffer;

      // Iterate over the pixels in the line
      for(; !iter_g.IsAtEndOfLine(); ++iter_g, grad_line++)
        {
        if(iter_g.CheckFixedMask())
          {
          // Reference to the gradient pointer
          GradientPixelType &grad_x = *grad_line;

          // Get the current histogram corners
          iter_g.PartialVolumeHistogramGradientSample(m_GradWeights, grad_x.GetDataPointer());
          }
        }
      }
    }

  else if(this->m_ComputeGradient && this->m_ComputeAffine)
    {
    GradientPixelType grad_x;
    typename Superclass::ThreadData &tds = this->m_ThreadData[threadId];

    int nvox = this->GetInput()->GetBufferedRegion().GetNumberOfPixels();

    // Iterate one more time through the voxels
    InterpType iter_g(this, this->GetMetricOutput(), outputRegionForThread);
    for(; !iter_g.IsAtEnd(); iter_g.NextLine())
      {
      // Iterate over the pixels in the line
      for(; !iter_g.IsAtEndOfLine(); ++iter_g)
        {
        if(iter_g.CheckFixedMask())
          {
          // Get the current histogram corners
          iter_g.PartialVolumeHistogramGradientSample(m_GradWeights, grad_x.GetDataPointer());

          // Add the gradient
          for(int i = 0, q = 0; i < ImageDimension; i++)
            {
            double v = grad_x[i] / nvox;
            tds.gradient[q++] += v;
            for(int j = 0; j < ImageDimension; j++)
              tds.gradient[q++] += v * iter_g.GetIndex()[j];
            }
          }
        }
      }
    }
}















template <class TInputImage, class TOutputImage>
MutualInformationPreprocessingFilter<TInputImage, TOutputImage>
::MutualInformationPreprocessingFilter()
{
  m_Barrier = itk::Barrier::New();

  m_LowerQuantile = 0.0;
  m_UpperQuantile = 0.99;

  m_NoRemapping = false;
}

template <class TInputImage, class TOutputImage>
void
MutualInformationPreprocessingFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();
  this->GetOutput()->SetNumberOfComponentsPerPixel(this->GetInput()->GetNumberOfComponentsPerPixel());
}

template <class TInputImage, class TOutputImage>
void
MutualInformationPreprocessingFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  m_ThreadData.clear();
  m_ThreadData.resize(this->GetNumberOfThreads());

  // Code to determine the actual number of threads used below
  itk::ThreadIdType nbOfThreads = this->GetNumberOfThreads();
  if ( itk::MultiThreader::GetGlobalMaximumNumberOfThreads() != 0 )
    {
    nbOfThreads = vnl_math_min( this->GetNumberOfThreads(), itk::MultiThreader::GetGlobalMaximumNumberOfThreads() );
    }

  typename TOutputImage::RegionType splitRegion;  // dummy region - just to call
                                                  // the following method
  nbOfThreads = this->SplitRequestedRegion(0, nbOfThreads, splitRegion);


  m_Barrier = itk::Barrier::New();
  m_Barrier->Initialize(nbOfThreads);

  m_LowerQuantileValues.resize(this->GetInput()->GetNumberOfComponentsPerPixel());
  m_UpperQuantileValues.resize(this->GetInput()->GetNumberOfComponentsPerPixel());
}

template <class TInputImage, class TOutputImage>
void
MutualInformationPreprocessingFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType &outputRegionForThread, itk::ThreadIdType threadId)
{
  // Determine the size of the heap
  int total_pixels = this->GetInput()->GetBufferedRegion().GetNumberOfPixels();
  int heap_size_upper = 1 + (int)((1.0 - m_UpperQuantile) * total_pixels);
  int heap_size_lower = 1 + (int)(m_LowerQuantile * total_pixels);
  int line_length = outputRegionForThread.GetSize(0);

  // Thread data for this thread
  ThreadData &td = m_ThreadData[threadId];

  typedef itk::ImageLinearConstIteratorWithIndex<InputImageType> IterBase;
  typedef IteratorExtender<IterBase> Iterator;

  // Iterate over each component
  int ncomp = this->GetInput()->GetNumberOfComponentsPerPixel();
  for(int k = 0; k < ncomp; k++)
    {
    // Initialize the two heaps
    td.heap_lower = LowerHeap();
    td.heap_upper = UpperHeap();

    // Build up the heaps
    for(Iterator it(this->GetInput(), outputRegionForThread); !it.IsAtEnd(); it.NextLine())
      {
      // Get a pointer to the start of the line
      const InputComponentType *line = it.GetPixelPointer(this->GetInput()) + k;

      // Iterate over the line
      for(int p = 0; p < line_length; p++, line+=ncomp)
        {
        InputComponentType v = *line;
        heap_lower_push(td.heap_lower, heap_size_lower, v);
        heap_upper_push(td.heap_upper, heap_size_upper, v);
        }
      }

    // Wait for the threads to synchronize
    m_Barrier->Wait();

    // The main thread combines all the priority queues
    if(threadId == 0)
      {
      // Combine the priority queues
      for(int q = 1; q < this->GetNumberOfThreads(); q++)
        {
        ThreadData &tdq = m_ThreadData[q];
        while(!tdq.heap_lower.empty())
          {
          InputComponentType v = tdq.heap_lower.top();
          heap_lower_push(td.heap_lower, heap_size_lower, v);
          tdq.heap_lower.pop();
          }

        while(!tdq.heap_upper.empty())
          {
          InputComponentType v = tdq.heap_upper.top();
          heap_upper_push(td.heap_upper, heap_size_upper, v);
          tdq.heap_upper.pop();
          }
        }

      // Get the quantile values
      m_UpperQuantileValues[k] = td.heap_upper.top();
      m_LowerQuantileValues[k] = td.heap_lower.top();
      }

    // Wait for all threads to catch up
    m_Barrier->Wait();

    // Continue if no remapping requested
    if(m_NoRemapping)
      continue;

    // Compute the scale and shift
    double scale = m_Bins * 1.0 / (m_UpperQuantileValues[k] - m_LowerQuantileValues[k]);
    double shift = m_LowerQuantileValues[k] * scale;

    // Now each thread remaps the intensities into the quantile range
    for(Iterator it(this->GetInput(), outputRegionForThread); !it.IsAtEnd(); it.NextLine())
      {
      // Get a pointer to the start of the line
      const InputComponentType *line = it.GetPixelPointer(this->GetInput()) + k;
      OutputComponentType *out_line = it.GetPixelPointer(this->GetOutput()) + k;

      // Iterate over the line
      for(int p = 0; p < line_length; p++, line+=ncomp, out_line+=ncomp)
        {
        int bin = (int) (*line * scale - shift);
        if(bin < 0)
          *out_line = 0;
        else if(bin >= m_Bins)
          *out_line = m_Bins - 1;
        else
          *out_line = bin;
        }
      }
    } // loop over components
}




#endif // MULTICOMPONENTMUTUALINFOIMAGEMETRIC_TXX
