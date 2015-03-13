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
#include "OneDimensionalInPlaceAccumulateFilter.h"
#include "itkUnaryFunctorImageFilter.h"

#include "LinearTransformToWarpFilter.h"

#include "itkImageFileWriter.h"

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

#include <vnl/vnl_random.h>

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::BuildCompositeImages(bool add_noise)
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

        // Add some noise to the images
        if(add_noise)
          {
          // TODO: remove this or make it optional
          vnl_random randy;
          for(long i = 0; i < lFixed->GetPixelContainer()->Size(); i++)
            lFixed->GetBufferPointer()[i] += randy.normal();
          for(long i = 0; i < lMoving->GetPixelContainer()->Size(); i++)
            lMoving->GetBufferPointer()[i] += randy.normal();
          }


        // Compute the gradient of the moving image
        //typename VectorImageType::Pointer gradMoving = VectorImageType::New();
        //LDDMMType::alloc_vimg(gradMoving, lMoving);
        //LDDMMType::image_gradient(lMoving, gradMoving);

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
          m_MovingComposite[i]->SetNumberOfComponentsPerPixel(m_Weights.size());
          m_MovingComposite[i]->SetRegions(lMoving->GetBufferedRegion());
          m_MovingComposite[i]->Allocate();
          }

        // Pack the data into the fixed and moving composite images
        this->PlaceIntoComposite(lFixed, m_FixedComposite[i], off_fixed);
        this->PlaceIntoComposite(lMoving, m_MovingComposite[i], off_moving);
        }

      // Update the offsets
      off_fixed++;
      off_moving++;
      }
    }

  // Set up the mask pyramid
  m_GradientMaskComposite.resize(m_PyramidFactors.size(), NULL);
  if(m_GradientMaskImage)
    {
    for(int i = 0; i < m_PyramidFactors.size(); i++)
      {
      // Downsample the image to the right pyramid level
      if (m_PyramidFactors[i] == 1)
        {
        m_GradientMaskComposite[i] = m_GradientMaskImage;
        }
      else
        {
        m_GradientMaskComposite[i] = FloatImageType::New();
        LDDMMType::img_downsample(m_GradientMaskImage, m_GradientMaskComposite[i], m_PyramidFactors[i]);
        }
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
vnl_vector<double>
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeOpticalFlowField(int level,
                          VectorImageType *def,
                          FloatImageType *out_metric,
                          VectorImageType *out_gradient,
                          double result_scaling)
{
  typedef DefaultMultiComponentImageMetricTraits<TFloat, VDim> TraitsType;
  typedef MultiImageOpticalFlowImageFilter<TraitsType> FilterType;

  typename FilterType::Pointer filter = FilterType::New();

  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for(int i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i] * result_scaling;

  // Run the filter
  filter->SetFixedImage(m_FixedComposite[level]);
  filter->SetMovingImage(m_MovingComposite[level]);
  filter->SetDeformationField(def);
  filter->SetWeights(wscaled);
  filter->SetComputeGradient(true);
  filter->GetMetricOutput()->Graft(out_metric);
  filter->GetGradientOutput()->Graft(out_gradient);
  filter->Update();

  // Get the vector of the normalized metrics
  return filter->GetAllMetricValues();
}

#undef DUMP_NCC
// #define DUMP_NCC 1


template <class TFloat, unsigned int VDim>
double
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeNCCMetricImage(int level,
                        VectorImageType *def,
                        const SizeType &radius,
                        FloatImageType *out_metric,
                        VectorImageType *out_gradient,
                        double result_scaling)
{
  typedef DefaultMultiComponentImageMetricTraits<TFloat, VDim> TraitsType;
  typedef MultiComponentNCCImageMetric<TraitsType> FilterType;

  typename FilterType::Pointer filter = FilterType::New();

  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for(int i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i] * result_scaling;

  // Allocate a working image
  if(m_NCCWorkingImage.IsNull())
    m_NCCWorkingImage = MultiComponentImageType::New();

  // Run the filter
  filter->SetFixedImage(m_FixedComposite[level]);
  filter->SetMovingImage(m_MovingComposite[level]);
  filter->SetDeformationField(def);
  filter->SetWeights(wscaled);
  filter->SetComputeGradient(true);
  filter->GetMetricOutput()->Graft(out_metric);
  filter->GetGradientOutput()->Graft(out_gradient);
  filter->SetRadius(radius);
  filter->SetWorkingImage(m_NCCWorkingImage);
  filter->Update();

  // Get the vector of the normalized metrics
  return filter->GetMetricValue();
}



template <class TFloat, unsigned int VDim>
double
MultiImageOpticalFlowHelper<TFloat, VDim>
::ComputeAffineMatchAndGradient(int level,
    LinearTransformType *tran,
    FloatImageType *wrkMetric,
    FloatImageType *wrkMask,
    VectorImageType *wrkGradMetric,
    VectorImageType *wrkGradMask,
    VectorImageType *wrkPhi,
    LinearTransformType *grad)
{
  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for(int i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i];

  // Create a deformation field from the affine transform
  typedef LinearTransformToWarpFilter<
      MultiComponentImageType, VectorImageType, LinearTransformType> WarpFilter;
  typename WarpFilter::Pointer warp_source = WarpFilter::New();
  warp_source->SetInput(m_FixedComposite[level]);
  warp_source->SetTransform(tran);
  warp_source->GraftOutput(wrkPhi);

  // Set up the optical flow computation
  typedef DefaultMultiComponentImageMetricTraits<TFloat, VDim> TraitsType;
  typedef MultiImageOpticalFlowImageFilter<TraitsType> MetricType;
  typename MetricType::Pointer metric = MetricType::New();

  metric->SetFixedImage(m_FixedComposite[level]);
  metric->SetMovingImage(m_MovingComposite[level]);
  metric->SetWeights(wscaled);
  metric->SetDeformationField(warp_source->GetOutput());

  metric->GetMetricOutput()->Graft(wrkMetric);

  metric->SetComputeMovingDomainMask(true);
  metric->GetMovingDomainMaskOutput()->Graft(wrkMask);

  if(grad)
    {
    metric->SetComputeGradient(true);
    metric->GetGradientOutput()->Graft(wrkGradMetric);
    metric->GetMovingDomainMaskGradientOutput()->Graft(wrkGradMask);
    }

  // Use finite differences
  typedef MultiImageAffineMetricFilter<TraitsType> AffineMetricType;
  typename AffineMetricType::Pointer affine_metric = AffineMetricType::New();

  // Run the filter
  affine_metric->SetMetricImage(metric->GetMetricOutput());
  affine_metric->SetMovingDomainMaskImage(metric->GetMovingDomainMaskOutput());

  // TODO: only if gradient!
  if(grad)
    {
    /*
    typedef itk::ImageFileWriter<FloatImageType> IW;
    typename IW::Pointer iw = IW::New();
    iw->SetFileName("dump_mask.nii.gz");
    iw->SetInput(metric->GetMovingDomainMaskOutput());
    iw->Update();

    iw->SetFileName("dump_metric.nii.gz");
    iw->SetInput(metric->GetMetricOutput());
    iw->Update();

    typedef itk::ImageFileWriter<VectorImageType> GW;
    typename GW::Pointer gw = GW::New();
    gw->SetFileName("dump_mask_grad.mha");
    gw->SetInput(metric->GetMovingDomainMaskGradientOutput());
    gw->Update();

    gw->SetFileName("dump_metric_grad.mha");
    gw->SetInput(metric->GetGradientOutput());
    gw->Update();

    gw->SetFileName("dump_phi.mha");
    gw->SetInput(wrkPhi);
    gw->Update();

    exit(-1);
    */

    affine_metric->SetComputeGradient(true);
    affine_metric->SetGradientImage(metric->GetGradientOutput());
    affine_metric->SetMovingDomainMaskGradientImage(metric->GetMovingDomainMaskGradientOutput());
    }

  affine_metric->Update();

  // Process the results
  if(grad)
    {
    grad->SetMatrix(affine_metric->GetMetricGradient()->GetMatrix());
    grad->SetOffset(affine_metric->GetMetricGradient()->GetOffset());
    }

  return affine_metric->GetMetricValue();

}

template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::AffineToField(LinearTransformType *tran, VectorImageType *def)
{
  // TODO: convert this to a filter
  typedef itk::ImageLinearIteratorWithIndex<VectorImageType> IterBase;
  typedef IteratorExtender<IterBase> Iter;
  Iter it(def, def->GetBufferedRegion());
  it.SetDirection(0);

  for(; !it.IsAtEnd(); it.NextLine())
    {
    // Get the pointer to the begin of line
    VectorType *ptr = const_cast<VectorType *>(it.GetPosition());
    VectorType *ptr_end = ptr + def->GetBufferedRegion().GetSize(0);

    // Get the initial index
    typename LinearTransformType::InputPointType pt;
    for(int k = 0; k < VDim; k++)
      pt[k] = it.GetIndex()[k];

    for(; ptr < ptr_end; ++ptr, ++pt[0])
      {
      // Apply transform to the index. TODO: this is stupid, just use an offset
      typename LinearTransformType::OutputPointType pp = tran->TransformPoint(pt);
      for(int k = 0; k < VDim; k++)
        (*ptr)[k] = pp[k] - pt[k];
      }
    }
}


template <class TInputImage, class TOutputImage, class TFunctor>
class UnaryPositionBasedFunctorImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:

  typedef UnaryPositionBasedFunctorImageFilter<TInputImage,TOutputImage,TFunctor> Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( UnaryPositionBasedFunctorImageFilter, itk::ImageToImageFilter )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension );

  void SetFunctor(const TFunctor &f) { this->m_Functor = f; }

protected:
  UnaryPositionBasedFunctorImageFilter() {}
  ~UnaryPositionBasedFunctorImageFilter() {}

  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                    itk::ThreadIdType threadId)
  {
    typedef itk::ImageRegionConstIteratorWithIndex<TInputImage> InputIter;
    InputIter it_in(this->GetInput(), outputRegionForThread);

    typedef itk::ImageRegionIterator<TOutputImage> OutputIter;
    OutputIter it_out(this->GetOutput(), outputRegionForThread);

    for(; !it_out.IsAtEnd(); ++it_out, ++it_in)
      {
      it_out.Set(m_Functor(it_in.Get(), it_in.GetIndex()));
      }
  }

  TFunctor m_Functor;
};

template <class TWarpImage>
struct VoxelToPhysicalWarpFunctor
{
  typedef itk::ImageBase<TWarpImage::ImageDimension> ImageBaseType;
  typedef typename TWarpImage::PixelType VectorType;
  typedef itk::Index<TWarpImage::ImageDimension> IndexType;

  VectorType operator()(const VectorType &v, const IndexType &pos)
  {
    // Get the physical point for the tail of the arrow
    typedef itk::ContinuousIndex<double, TWarpImage::ImageDimension> CIType;
    typedef typename TWarpImage::PointType PtType;

    CIType ia, ib;
    PtType pa, pb;
    for(int i = 0; i < TWarpImage::ImageDimension; i++)
      {
      ia[i] = pos[i];
      ib[i] = pos[i] + v[i];
      }

    m_Warp->TransformContinuousIndexToPhysicalPoint(ia, pa);
    m_MovingSpace->TransformContinuousIndexToPhysicalPoint(ib, pb);

    VectorType y;
    for(int i = 0; i < TWarpImage::ImageDimension; i++)
      y[i] = pb[i] - pa[i];

    return y;
  }


  TWarpImage *m_Warp;
  ImageBaseType *m_MovingSpace;
};


template <class TFloat, unsigned int VDim>
void
MultiImageOpticalFlowHelper<TFloat, VDim>
::VoxelWarpToPhysicalWarp(int level, VectorImageType *warp, VectorImageType *result)
{
  typedef VoxelToPhysicalWarpFunctor<VectorImageType> Functor;
  typedef UnaryPositionBasedFunctorImageFilter<VectorImageType,VectorImageType,Functor> Filter;
  Functor functor;
  functor.m_Warp = warp;
  functor.m_MovingSpace = this->GetMovingReferenceSpace(level);

  typename Filter::Pointer filter = Filter::New();
  filter->SetFunctor(functor);
  filter->SetInput(warp);
  filter->GraftOutput(result);
  filter->Update();
}

#endif
