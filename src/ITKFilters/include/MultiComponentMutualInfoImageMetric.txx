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
#include "MultiComponentQuantileBasedNormalizationFilter.h"
#include <functional>


/**
 * Implementation of the Normalized Mutual Information (Studholme) method
 *
 * Metric = (H(M) + H(F)) / H(M,F)
 */
template <class TReal>
TReal
NormalizedMutualInformationMetricFunction<TReal>
::compute(int n_bins,
          const vnl_matrix<TReal> &mat_Pfm,
          const vnl_vector<TReal> &mat_Pf,
          const vnl_vector<TReal> &mat_Pm,
          vnl_matrix<TReal> *gradWeights)
{
  // We need the joint and marginal entropies for the calculation
  TReal Hfm = 0, Hf = 0, Hm = 0;

  // Simple case - no gradient
  if(!gradWeights)
    {
    for(int i = 1; i < n_bins; i++)
      {
      TReal Pf = mat_Pf(i);
      TReal Pm = mat_Pm(i);

      if(Pf > 0)
        Hf += Pf * log(Pf);

      if(Pm > 0)
        Hm += Pm * log(Pm);

      for(int j = 1; j < n_bins; j++)
        {
        TReal Pfm = mat_Pfm(i, j);
        if(Pfm > 0)
          Hfm += Pfm * log(Pfm);
        }
      }

    return (Hf + Hm) / Hfm;
    }

  else
    {
    // Allocate vectors to hold log(Pf), log(Pm)
    vnl_vector<TReal> log_Pf(n_bins, 0.0), log_Pm(n_bins, 0.0);

    for(int i = 1; i < n_bins; i++)
      {
      TReal Pf = mat_Pf(i);
      TReal Pm = mat_Pm(i);

      if(Pf > 0)
        {
        log_Pf(i) = log(Pf);
        Hf += Pf * log_Pf(i);
        }

      if(Pm > 0)
        {
        log_Pm(i) = log(Pm);
        Hm += Pm * log_Pm(i);
        }

      for(int j = 1; j < n_bins; j++)
        {
        TReal Pfm = mat_Pfm(i, j);
        if(Pfm > 0)
          {
          TReal log_Pfm = log(Pfm);
          Hfm += Pfm * log_Pfm;
          (*gradWeights)(i, j) = log_Pfm; // store for future use
          }
        }
      }

    // Compute the metric
    TReal metric = (Hf + Hm) / Hfm;

    // Compute the gradient
    for(int i = 1; i < n_bins; i++)
      {
      for(int j = 1; j < n_bins; j++)
        {
        TReal Pfm = mat_Pfm(i, j);
        if(Pfm > 0)
          {
          // Reuse the log
          TReal log_Pfm = (*gradWeights)(i, j);
          (*gradWeights)(i,j) = (2 + log_Pf(i) + log_Pm(j) - metric * (log_Pfm + 1)) / Hfm;
          }
        else
          {
          (*gradWeights)(i,j) = 0.0;
          }
        }
      }

    // Return the metric
    return metric;
    }
}

/**
 * Implementation of the standard Mutual Information method
 *
 * Metric = H(M,F) - H(M) - H(F)
 */
template <class TReal>
TReal
StandardMutualInformationMetricFunction<TReal>
::compute(int n_bins,
          const vnl_matrix<TReal> &mat_Pfm,
          const vnl_vector<TReal> &mat_Pf,
          const vnl_vector<TReal> &mat_Pm,
          vnl_matrix<TReal> *gradWeights)
{
  TReal metric = 0;
  for(int bf = 1; bf < n_bins; bf++)
    {
    for(int bm = 1; bm < n_bins; bm++)
      {
      TReal Pfm = mat_Pfm(bf, bm);
      TReal Pf = mat_Pf(bf);
      TReal Pm = mat_Pm(bm);

      if(Pfm > 0)
        {
        // This expression is actually correct for computing H(I,J) - (H(I) + H(J))
        double q = log(Pfm / (Pf * Pm));
        double v = Pfm * q;
        metric += v;

        // If computing the gradient, also compute the additional weight information
        if(gradWeights)
          {
          (*gradWeights)[bf][bm] = q - 1;
          }
        }
      }
    }

  return metric;
}


template <class TMetricTraits>
void
MultiComponentMutualInfoImageMetric<TMetricTraits>
::BeforeThreadedGenerateData()
{
  Superclass::BeforeThreadedGenerateData();

  int ncomp = this->GetFixedImage()->GetNumberOfComponentsPerPixel();

  // Initialize the gradient matrices
  if(this->m_ComputeGradient)
    m_GradWeights.resize(ncomp, vnl_matrix<RealType>(m_Bins, m_Bins, 0.0));

}


template <class TMetricTraits>
void
MultiComponentMutualInfoImageMetric<TMetricTraits>
::GenerateData()
{
  // Standard stuff done before splitting into threads
  this->AllocateOutputs();
  this->BeforeThreadedGenerateData();

  // Get the number of components
  int ncomp = this->GetFixedImage()->GetNumberOfComponentsPerPixel();

  // Create an iterator specialized for going through metrics
  typedef MultiComponentMetricWorker<TMetricTraits, MetricImageType> InterpType;

  // Initially, I am implementing this as a two-pass filter. On the first pass, the joint
  // histogram is computed without the gradient. On the second pass, the gradient is computed.
  // The inefficiency of this implementation is that the interpolation code is being called
  // twice. The only way I see to avoid this is to store the results of each interpolation in
  // an intermediate working image, but I am not sure how much one would save from that!

  // Importantly, the input images are rescaled to the range 1..nBins. This means that the
  // 0 row and 0 column of the histogram are reserved for outside values, or in other words
  // outside values are treated differently from zero. This is important for the computation
  // of the overlap-invariant metrics.

  // Initialize the per-component histogram array
  m_Histograms.resize(ncomp, Histogram(m_Bins));

  // Image-wide histogram accumulator
  std::mutex hist_mutex;

  // Use the new ITK5 code for parallelization. The result of this will be to compute the
  // histogram of the entire image in hist_pooled
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
  mt->ParallelizeImageRegion<Self::ImageDimension>(
        this->GetOutput()->GetBufferedRegion(),
        [this,&ncomp,&hist_mutex](const OutputImageRegionType &region)
    {
    // This is the histogram accumulator local to this thread
    HistogramAccumType hist_local(ncomp, vnl_matrix<RealType>(m_Bins, m_Bins, 0.0));

    // This is the iterator for the image region
    InterpType iter(this, this->GetMetricOutput(), region);

    // Iterate over the lines
    for(; !iter.IsAtEnd(); iter.NextLine())
      {
      // Iterate over the pixels in the line
      for(; !iter.IsAtEndOfLine(); ++iter)
        {
        // Get the current histogram corners
        if(iter.CheckFixedMask())
          iter.PartialVolumeHistogramSample(hist_local);
        }
      }

    // Use a mutex to update the combined histogram
    std::lock_guard<std::mutex> guard(hist_mutex);
    for(int c = 0; c < ncomp; c++)
      {
      // We only capture the non-outside histogram bin values
      for (unsigned bf = 1; bf < m_Bins; bf++)
        for(unsigned bm = 1; bm < m_Bins; bm++)
          this->m_Histograms[c].Pfm(bf,bm) += hist_local[c](bf,bm);
      }
    }, nullptr);

  // All procesing is separate for each component
  for(int c = 0; c < ncomp; c++)
    {
    // The histogram for this component
    Histogram &hc = m_Histograms[c];

    // When computing the empirical joint probability, we will ignore outside values.
    // We need multiple passes through the histogram to calculate the emprirical prob.

    // First pass, add thread data and compute the sum of all non-outside histogram bin balues
    double hist_sum = 0.0;
    for (unsigned bf = 1; bf < m_Bins; bf++)
      for(unsigned bm = 1; bm < m_Bins; bm++)
        hist_sum += hc.Pfm(bf,bm);

    // Second pass, normalize the entries and compute marginals
    for (unsigned bf = 1; bf < m_Bins; bf++)
      {
      for(unsigned bm = 1; bm < m_Bins; bm++)
        {
        // Reference to the joint probability entry
        RealType &Pfm = hc.Pfm(bf,bm);

        // Normalize to make a probability
        Pfm /= hist_sum;

        // Add up the marginals
        hc.Pf[bf] += Pfm;
        hc.Pm[bm] += Pfm;
        }
      }

    // Third pass: compute the mutual information for this component and overall
    auto &m_comp = this->m_AccumulatedData.comp_metric[c];
    auto &m_total = this->m_AccumulatedData.metric;

    // Compute the metric and gradient for this component using the emprical probabilities
    if(this->m_ComputeNormalizedMutualInformation)
      m_comp = NormalizedMutualInformationMetricFunction<RealType>::compute(
            m_Bins, hc.Pfm, hc.Pf, hc.Pm, this->m_ComputeGradient ? &this->m_GradWeights[c] : NULL);
    else
      m_comp = StandardMutualInformationMetricFunction<RealType>::compute(
            m_Bins, hc.Pfm, hc.Pf, hc.Pm, this->m_ComputeGradient ? &this->m_GradWeights[c] : NULL);

    // Scale the gradient weights for this component
    m_comp *= this->m_Weights[c];
    if(this->m_ComputeGradient)
      this->m_GradWeights[c] *= this->m_Weights[c];

    m_total += m_comp;

    if(this->m_ComputeGradient)
      {
      // The gradient is currently relative to emprical probabilities. Convert it to gradient
      // in terms of the bin counts
      double grad_weights_dot_Pfm = 0.0;

      for (unsigned bf = 1; bf < m_Bins; bf++)
        {
        for(unsigned bm = 1; bm < m_Bins; bm++)
          {
          double Pfm = hc.Pfm(bf, bm);
          if(Pfm > 0)
            grad_weights_dot_Pfm += m_GradWeights[c][bf][bm] * Pfm;
          }
        }

      for (unsigned bf = 1; bf < m_Bins; bf++)
        {
        for(unsigned bm = 1; bm < m_Bins; bm++)
          {
          m_GradWeights[c][bf][bm] = (m_GradWeights[c][bf][bm] - grad_weights_dot_Pfm) / hist_sum;
          }
        }
      }

    } // loop over components

  // The last thing is to set the normalizing constant to 1
  this->m_AccumulatedData.mask = 1.0;

  // The second threaded pass is used for gradient computation
  mt->ParallelizeImageRegion<Self::ImageDimension>(
        this->GetOutput()->GetBufferedRegion(),
        [this,&ncomp,&hist_mutex](const OutputImageRegionType &region)
    {
    // At this point, we should be computing the gradient using the probability values computed above
    if(this->m_ComputeGradient && !this->m_ComputeAffine)
      {
      GradientPixelType *grad_buffer = this->GetDeformationGradientOutput()->GetBufferPointer();

      // Iterate one more time through the voxels
      InterpType iter_g(this, this->GetMetricOutput(), region);
      for(; !iter_g.IsAtEnd(); iter_g.NextLine())
        {
        // Get the output gradient pointer at the beginning of this line
        GradientPixelType *grad_line = iter_g.GetOffsetInPixels() + grad_buffer;

        // Iterate over the pixels in the line
        GradientPixelType grad_x;
        for(; !iter_g.IsAtEndOfLine(); ++iter_g, grad_line++)
          {
          if(iter_g.CheckFixedMask())
            {
            // Get the current histogram corners
            iter_g.PartialVolumeHistogramGradientSample(m_GradWeights, grad_x.GetDataPointer());

            // Accumulate in the gradient output
            *grad_line += grad_x;
            }
          }
        }
      }

    else if(this->m_ComputeGradient && this->m_ComputeAffine)
      {
      GradientPixelType grad_x;

      // Keep track of our thread's gradient contribution
      vnl_vector<double> grad_local(Superclass::ThreadAccumulatedData::GradientSize, 0.);

      // Iterate one more time through the voxels
      InterpType iter_g(this, this->GetMetricOutput(), region);
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
              // double v = grad_x[i] / nvox;
              double v = grad_x[i];
              grad_local[q++] += v;
              for(int j = 0; j < ImageDimension; j++)
                grad_local[q++] += v * iter_g.GetIndex()[j];
              }
            }
          }
        }

      // Add the local gradient to the overall gradient
      std::lock_guard<std::mutex> guard(this->m_AccumulatedData.mutex);
      this->m_AccumulatedData.gradient += grad_local;
      }
    }, nullptr);

  this->AfterThreadedGenerateData();
}




#endif // MULTICOMPONENTMUTUALINFOIMAGEMETRIC_TXX
