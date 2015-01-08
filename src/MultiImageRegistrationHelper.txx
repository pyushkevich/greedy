/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: SimpleWarpImageFilter.txx,v $
  Language:  C++
  Date:      $Date: 2009-10-29 11:19:10 $
  Version:   $Revision: 1.34 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __MultiImageRegistrationHelper_txx
#define __MultiImageRegistrationHelper_txx
#include "MultiImageRegistrationHelper.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkNumericTraits.h"
#include "itkContinuousIndex.h"
#include "vnl/vnl_math.h"
#include "lddmm_data.h"
#include "MultiImageAffineMSDMetricFilter.h"
#include "MultiImageOpticalFlowImageFilter.h"
#include "itkVectorIndexSelectionCastImageFilter.h"

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::SetDefaultPyramidFactors(int n_levels)
{
  for(int i = n_levels-1; i>=0; --i)
    m_PyramidFactors.push_back(1 << i);
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::SetPyramidFactors(const PyramidFactorsType &factors)
{
  m_PyramidFactors = factors;
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::AddImagePair(MultiComponentImageType *fixed, MultiComponentImageType *moving, double weight)
{
  // Collect the weights
  for(int i = 0; i < fixed->GetNumberOfComponentsPerPixel(); i++)
    m_Weights.push_back(weight);

  // Store the images
  m_Fixed.push_back(fixed);
  m_Moving.push_back(moving);
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::PlaceIntoComposite(FloatImageType *source, MultiComponentImageType *target, int offset)
{
  // We do this using a loop - no threading
  TFloat *src_ptr = source->GetPixelContainer()->GetBufferPointer();
  TFloat *trg_ptr = target->GetPixelContainer()->GetBufferPointer() + offset;

  int trg_comp = target->GetNumberOfComponentsPerPixel();

  int n_voxels = source->GetPixelContainer()->Size();
  TFloat *trg_end = trg_ptr + n_voxels * target->GetNumberOfComponentsPerPixel();

  while(trg_ptr < trg_end)
    {
    *trg_ptr = *src_ptr++;
    trg_ptr += trg_comp;
    }
}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::PlaceIntoComposite(VectorImageType *source, MultiComponentImageType *target, int offset)
{
  // We do this using a loop - no threading
  VectorType *src_ptr = source->GetPixelContainer()->GetBufferPointer();
  TFloat *trg_ptr = target->GetPixelContainer()->GetBufferPointer() + offset;

  int trg_skip = target->GetNumberOfComponentsPerPixel() - VDim;

  int n_voxels = source->GetPixelContainer()->Size();
  TFloat *trg_end = trg_ptr + n_voxels * target->GetNumberOfComponentsPerPixel();

  while(trg_ptr < trg_end)
    {
    const VectorType &vsrc = *src_ptr++;
    for(int k = 0; k < VDim; k++)
      *trg_ptr++ = vsrc[k];
    trg_ptr += trg_skip;
    }
}


template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::BuildCompositeImages()
{
  typedef LDDMMData<TFloat, VDim> LDDMMType;

  // Offsets into the composite images
  int off_fixed = 0, off_moving = 0;

  // Set up the composite images
  m_FixedComposite.resize(m_PyramidFactors.size());
  m_MovingComposite.resize(m_PyramidFactors.size());

  // Repeat for each of the input images
  for(int j = 0; j < m_Fixed.size(); j++)
    {
    // Repeat for each component
    for(int k = 0; k < m_Fixed[j]->GetNumberOfComponentsPerPixel(); k++)
      {
      // Extract the k-th image component from fixed and moving images
      typedef itk::VectorIndexSelectionCastImageFilter<MultiComponentImageType, FloatImageType> ExtractType;
      typename ExtractType::Pointer fltExtractFixed, fltExtractMoving;

      fltExtractFixed = ExtractType::New();
      fltExtractFixed->SetInput(m_Fixed[j]);
      fltExtractFixed->SetIndex(k);
      fltExtractFixed->Update();

      fltExtractMoving = ExtractType::New();
      fltExtractMoving->SetInput(m_Moving[j]);
      fltExtractMoving->SetIndex(k);
      fltExtractMoving->Update();

      // Compute the pyramid for this component
      for(int i = 0; i < m_PyramidFactors.size(); i++)
        {
        // Downsample the image to the right pyramid level
        typename FloatImageType::Pointer lFixed, lMoving;
        if (m_PyramidFactors[i] == 1)
          {
          lFixed = fltExtractFixed->GetOutput();
          lMoving = fltExtractMoving->GetOutput();
          }
        else
          {
          lFixed = FloatImageType::New();
          lMoving = FloatImageType::New();
          LDDMMType::img_downsample(fltExtractFixed->GetOutput(), lFixed, m_PyramidFactors[i]);
          LDDMMType::img_downsample(fltExtractMoving->GetOutput(), lMoving, m_PyramidFactors[i]);
          }

        // Compute the gradient of the moving image
        typename VectorImageType::Pointer gradMoving = VectorImageType::New();
        LDDMMType::alloc_vimg(gradMoving, lMoving);
        LDDMMType::image_gradient(lMoving, gradMoving);

        // Allocate the composite images if they have not been allocated
        if(j == 0 && k == 0)
          {
          m_FixedComposite[i] = MultiComponentImageType::New();
          m_FixedComposite[i]->CopyInformation(lFixed);
          m_FixedComposite[i]->SetNumberOfComponentsPerPixel(m_Weights.size());
          m_FixedComposite[i]->SetRegions(lFixed->GetBufferedRegion());
          m_FixedComposite[i]->Allocate();

          m_MovingComposite[i] = MultiComponentImageType::New();
          m_MovingComposite[i]->CopyInformation(lMoving);
          m_MovingComposite[i]->SetNumberOfComponentsPerPixel(m_Weights.size() * (1 + VDim));
          m_MovingComposite[i]->SetRegions(lMoving->GetBufferedRegion());
          m_MovingComposite[i]->Allocate();
          }

        // Pack the data into the fixed and moving composite images
        this->PlaceIntoComposite(lFixed, m_FixedComposite[i], off_fixed);
        this->PlaceIntoComposite(lMoving, m_MovingComposite[i], off_moving);
        this->PlaceIntoComposite(gradMoving, m_MovingComposite[i], off_moving + 1);
        }

      // Update the offsets
      off_fixed++;
      off_moving += (1 + VDim);
      }
    }
}

template <class TFloat, unsigned int VDim>
typename MultiImageOpticalFlowHelper<TFloat, VDim>::ImageBaseType *
MultiImageOpticalFlowHelper<TFloat, VDim>
::GetMovingReferenceSpace(int level)
{
  return m_MovingComposite[level];
}

template <class TFloat, unsigned int VDim>
typename MultiImageOpticalFlowHelper<TFloat, VDim>::ImageBaseType *
MultiImageOpticalFlowHelper<TFloat, VDim>
::GetReferenceSpace(int level)
{
  return m_FixedComposite[level];
}

template <class TFloat, unsigned int VDim>
double
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeOpticalFlowField(int level, VectorImageType *def, VectorImageType *result, double result_scaling)
{
  typedef MultiImageOpticalFlowImageFilter<
      MultiComponentImageType, VectorImageType, VectorImageType> FilterType;

  typename FilterType::Pointer filter = FilterType::New();

  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for(int i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i] * result_scaling;

  // Run the filter
  filter->SetFixedImage(m_FixedComposite[level]);
  filter->SetMovingImageAndGradient(m_MovingComposite[level]);
  filter->SetDeformationField(def);
  filter->SetWeights(wscaled);
  filter->GraftOutput(result);
  filter->Update();

  // Get the total energy
  return filter->GetMetricValue();
}

template <class TFloat, unsigned int VDim>
double
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeAffineMatchAndGradient(
    int level, LinearTransformType *tran,
    LinearTransformType *grad)
{
  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for(int i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i];

  // Use finite differences
  typedef MultiImageAffineMSDMetricFilter<MultiComponentImageType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();

  // Run the filter
  filter->SetFixedImage(m_FixedComposite[level]);
  filter->SetMovingImageAndGradient(m_MovingComposite[level]);
  filter->SetTransform(tran);
  filter->SetWeights(wscaled);
  filter->SetComputeGradient(grad != NULL);
  filter->Update();

  // Process the results
  if(grad)
    {
    grad->SetMatrix(filter->GetMetricGradient()->GetMatrix());
    grad->SetOffset(filter->GetMetricGradient()->GetOffset());
    }

  return filter->GetMetricValue();

  /*
  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for(int i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i];

  // Use finite differences
  typedef itk::MultiImageAffineMSDMetricFilter<MultiComponentImageType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();

  // Run the filter
  filter->SetFixedImage(m_FixedComposite[level]);
  filter->SetMovingImageAndGradient(m_MovingComposite[level]);
  filter->SetTransform(tran);
  filter->SetWeights(wscaled);
  filter->SetComputeGradient(false);
  filter->Update();

  double f0 = filter->GetMetricValue();

  // Compute finite differences
  if(grad)
    {
    vnl_vector<float> x(12), gradf(12);
    itk::flatten_affine_transform(tran, x.data_block());
    for(int k = 0; k < 12; k++)
      {
      double fk[2], eps = 1.0e-3;
      for(int q = 0; q < 2; q++)
        {
        typename LinearTransformType::Pointer tranq = LinearTransformType::New();
        vnl_vector<float> xq = x;
        xq[k] += (q == 0 ? -1 : 1) * eps;
        itk::unflatten_affine_transform(xq.data_block(), tranq.GetPointer());

        filter = FilterType::New();
        filter->SetFixedImage(m_FixedComposite[level]);
        filter->SetMovingImageAndGradient(m_MovingComposite[level]);
        filter->SetTransform(tranq);
        filter->SetWeights(wscaled);
        filter->SetComputeGradient(false);
        filter->Update();

        fk[q] = filter->GetMetricValue();
        }
      gradf[k] = (fk[1]-fk[0]) / (2.0 * eps);
      }
    itk::unflatten_affine_transform(gradf.data_block(), grad);
    }

  return f0;

  */
}


#endif
