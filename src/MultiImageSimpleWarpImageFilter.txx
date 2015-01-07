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
#ifndef __MultiImageSimpleWarpImageFilter_txx
#define __MultiImageSimpleWarpImageFilter_txx
#include "MultiImageSimpleWarpImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkNumericTraits.h"
#include "itkProgressReporter.h"
#include "itkContinuousIndex.h"
#include "vnl/vnl_math.h"
#include "lddmm_data.h"

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


#include "itkVectorIndexSelectionCastImageFilter.h"
#include "itkImageFileWriter.h"

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

  typedef itk::ImageFileWriter<MultiComponentImageType> Writer;
  typename Writer::Pointer writer = Writer::New();
  writer->SetInput(m_MovingComposite[0]);
  writer->SetFileName("composite.nii.gz");
  writer->Update();
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
  typedef itk::MultiImageOpticalFlowWarpTraits<
      MultiComponentImageType, VectorImageType, VectorImageType> TraitsType;
  typedef itk::MultiImageOpticalFlowImageFilter<
      MultiComponentImageType, VectorImageType, TraitsType> FilterType;

  typename FilterType::Pointer filter = FilterType::New();

  // Scale the weights by epsilon
  vnl_vector<float> wscaled(m_Weights.size());
  for(int i = 0; i < wscaled.size(); i++)
    wscaled[i] = m_Weights[i] * result_scaling;

  // Run the filter
  filter->SetFixedImage(m_FixedComposite[level]);
  filter->SetMovingImageAndGradient(m_MovingComposite[level]);
  // filter->SetDeformationField(def);
  filter->SetTransform(def);
  filter->SetWeights(wscaled);
  filter->GraftOutput(result);
  filter->Update();

  // Get the total energy
  return filter->GetSummaryResult()[0];
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

  if(grad)
    {
    typedef itk::MultiImageOpticalFlowAffineGradientTraits<
        MultiComponentImageType, VectorImageType> TraitsType;
    typedef itk::MultiImageOpticalFlowImageFilter<
        MultiComponentImageType, VectorImageType, TraitsType> FilterType;

    typename FilterType::Pointer filter = FilterType::New();

    // Run the filter
    filter->SetFixedImage(m_FixedComposite[level]);
    filter->SetMovingImageAndGradient(m_MovingComposite[level]);
    filter->SetTransform(tran);
    filter->SetWeights(wscaled);

    // TODO: stop the filter from allocating a image pointlessly!
    // filter->GraftOutput(result);
    filter->Update();

    // Process the results, scaling by -2
    itk::unflatten_affine_transform(filter->GetSummaryResult().data_block()+1, grad, -2.0);

    // Get the total energy
    return filter->GetSummaryResult()[0];
    }
  else
    {
    typedef itk::MultiImageOpticalFlowAffineObjectiveTraits<
        MultiComponentImageType, VectorImageType> TraitsType;
    typedef itk::MultiImageOpticalFlowImageFilter<
        MultiComponentImageType, VectorImageType, TraitsType> FilterType;

    typename FilterType::Pointer filter = FilterType::New();

    // Run the filter
    filter->SetFixedImage(m_FixedComposite[level]);
    filter->SetMovingImageAndGradient(m_MovingComposite[level]);
    filter->SetTransform(tran);
    filter->SetWeights(wscaled);
    // TODO: stop the filter from allocating a image pointlessly!

    // filter->GraftOutput(result);
    filter->Update();

    // Process the results
    return filter->GetSummaryResult()[0];
    }
}





namespace itk
{

/**
 * Default constructor.
 */

template <class TInputImage, class TOutputImage, class TTransformTraits>
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TTransformTraits>
::MultiImageOpticalFlowImageFilter()
{
  // Setup default values
  // m_DeformationScaling = 1.0;
}

/**
 * Standard PrintSelf method.
 */
template <class TInputImage, class TOutputImage, class TTransformTraits>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TTransformTraits>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

/**
 * Setup state of filter before multi-threading.
 * InterpolatorType::SetInputImage is not thread-safe and hence
 * has to be setup before ThreadedGenerateData
 */
template <class TInputImage, class TOutputImage, class TTransformTraits>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TTransformTraits>
::BeforeThreadedGenerateData()
{
  // Create the prototype results vector
  InputImageType *moving = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("moving"));
  int kMoving = moving->GetNumberOfComponentsPerPixel();
  int nResult = TTransformTraits::GetResultAccumSize(kMoving);
  m_SummaryResult = SummaryType(nResult, 0.0);

  // Clear the energy per thread array
  m_SummaryResultPerThread =
      std::vector<SummaryType>(this->GetNumberOfThreads(), m_SummaryResult);
}

/**
 * Setup state of filter after multi-threading.
 */
template <class TInputImage, class TOutputImage, class TTransformTraits>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TTransformTraits>
::AfterThreadedGenerateData()
{
  for(int i = 0; i < m_SummaryResultPerThread.size(); i++)
    m_SummaryResult += m_SummaryResultPerThread[i];
}


/**
  Trilinear interpolation - code borrowed from http://tog.acm.org/resources/GraphicsGems/gemsiv/trilerp.c
 */
#define INRANGE(X, Y, Z) ((X) >= 0 && (X) < xsize && (Y) >= 0 && (Y) < ysize && (Z) >= 0 && (Z) < zsize)
#define DENS(X, Y, Z, ptr, comp) (ptr + comp * ((X)+xsize*((Y)+ysize*(Z))))

template <class TFloat>
inline TFloat LERP(TFloat a, TFloat l, TFloat h)
{
  return l+((h-l)*a);
}

/*
template <class TInputImage, class TOutputImage, class TDeformationField>
double
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TDeformationField>
::OpticalFlowFastInterpolate(const Dispatch<3> &,
                             float *cix,
                             const InputComponentType *fixed_ptr,
                             const InputComponentType *moving_ptr,
                             OutputPixelType &outVector,
                             int *movSize,
                             int nComp,
                             const InputComponentType *def_value)
{
  int	x0, y0, z0, x1, y1, z1;
  const InputComponentType *dp, *d000, *d001, *d010, *d011, *d100, *d101, *d110, *d111;

  double fx, fy, fz;
  double dx00, dx01, dx10, dx11, dxy0, dxy1, dxyz;

  int xsize = movSize[0];
  int ysize = movSize[1];
  int zsize = movSize[2];

  x0 = floor(cix[0]); fx = cix[0] - x0;
  y0 = floor(cix[1]); fy = cix[1] - y0;
  z0 = floor(cix[2]); fz = cix[2] - z0;

  x1 = x0 + 1;
  y1 = y0 + 1;
  z1 = z0 + 1;

  if (x0 >= 0 && x1 < xsize &&
      y0 >= 0 && y1 < ysize &&
      z0 >= 0 && z1 < zsize)
    {
    dp = DENS(x0, y0, z0, moving_ptr, nComp);
    d000 = dp;
    d100 = dp+nComp;
    dp += xsize*nComp;
    d010 = dp;
    d110 = dp+nComp;
    dp += xsize*ysize*nComp;
    d011 = dp;
    d111 = dp+nComp;
    dp -= xsize*nComp;
    d001 = dp;
    d101 = dp+nComp;
    }
  else
    {
    d000 = INRANGE(x0, y0, z0) ? DENS(x0, y0, z0, moving_ptr, nComp) : def_value;
    d001 = INRANGE(x0, y0, z1) ? DENS(x0, y0, z1, moving_ptr, nComp) : def_value;
    d010 = INRANGE(x0, y1, z0) ? DENS(x0, y1, z0, moving_ptr, nComp) : def_value;
    d011 = INRANGE(x0, y1, z1) ? DENS(x0, y1, z1, moving_ptr, nComp) : def_value;
    d100 = INRANGE(x1, y0, z0) ? DENS(x1, y0, z0, moving_ptr, nComp) : def_value;
    d101 = INRANGE(x1, y0, z1) ? DENS(x1, y0, z1, moving_ptr, nComp) : def_value;
    d110 = INRANGE(x1, y1, z0) ? DENS(x1, y1, z0, moving_ptr, nComp) : def_value;
    d111 = INRANGE(x1, y1, z1) ? DENS(x1, y1, z1, moving_ptr, nComp) : def_value;
    }

  // Output vector
  double Vx = 0.0, Vy = 0.0, Vz = 0.0;

  // Output value
  double Tval = 0.0;

  // Weight array
  float *weight = m_Weights.data_block();

  // Interpolate each component
  for(int iComp = 0; iComp < nComp; iComp+=4)
    {
    double M, Mx, My, Mz;

    // TODO: parallelize this using SSD

    // Interpolate first component
    dx00 = LERP(fx, *d000++, *d100++);
    dx01 = LERP(fx, *d001++, *d101++);
    dx10 = LERP(fx, *d010++, *d110++);
    dx11 = LERP(fx, *d011++, *d111++);
    dxy0 = LERP(fy, dx00, dx10);
    dxy1 = LERP(fy, dx01, dx11);
    M = LERP(fz, dxy0, dxy1);

    dx00 = LERP(fx, *d000++, *d100++);
    dx01 = LERP(fx, *d001++, *d101++);
    dx10 = LERP(fx, *d010++, *d110++);
    dx11 = LERP(fx, *d011++, *d111++);
    dxy0 = LERP(fy, dx00, dx10);
    dxy1 = LERP(fy, dx01, dx11);
    Mx = LERP(fz, dxy0, dxy1);

    dx00 = LERP(fx, *d000++, *d100++);
    dx01 = LERP(fx, *d001++, *d101++);
    dx10 = LERP(fx, *d010++, *d110++);
    dx11 = LERP(fx, *d011++, *d111++);
    dxy0 = LERP(fy, dx00, dx10);
    dxy1 = LERP(fy, dx01, dx11);
    My = LERP(fz, dxy0, dxy1);

    dx00 = LERP(fx, *d000++, *d100++);
    dx01 = LERP(fx, *d001++, *d101++);
    dx10 = LERP(fx, *d010++, *d110++);
    dx11 = LERP(fx, *d011++, *d111++);
    dxy0 = LERP(fy, dx00, dx10);
    dxy1 = LERP(fy, dx01, dx11);
    Mz = LERP(fz, dxy0, dxy1);

    // Compute the difference
    double del = (*fixed_ptr++) - M;
    double delw = (*weight++) * del;
    Vx += delw * Mx;
    Vy += delw * My;
    Vz += delw * Mz;
    Tval += delw * del;
    }

  // Store the output
  outVector[0] = Vx;
  outVector[1] = Vy;
  outVector[2] = Vz;

  // What to return?
  return Tval;
}
*/

template <class TInputImage, class TOutputImage, class TTransformTraits>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TTransformTraits>
::OpticalFlowFastInterpolate(const Dispatch<3> &,
                             const InputComponentType *moving_ptr,
                             int nComp, int stride, int *movSize,
                             const InputComponentType *def_value,
                             float *cix,
                             InputComponentType *out)
{
  int	x0, y0, z0, x1, y1, z1;
  const InputComponentType *dp, *d000, *d001, *d010, *d011, *d100, *d101, *d110, *d111;

  double fx, fy, fz;
  double dx00, dx01, dx10, dx11, dxy0, dxy1, dxyz;

  int xsize = movSize[0];
  int ysize = movSize[1];
  int zsize = movSize[2];

  x0 = floor(cix[0]); fx = cix[0] - x0;
  y0 = floor(cix[1]); fy = cix[1] - y0;
  z0 = floor(cix[2]); fz = cix[2] - z0;

  x1 = x0 + 1;
  y1 = y0 + 1;
  z1 = z0 + 1;

  if (x0 >= 0 && x1 < xsize &&
      y0 >= 0 && y1 < ysize &&
      z0 >= 0 && z1 < zsize)
    {
    dp = DENS(x0, y0, z0, moving_ptr, nComp);
    d000 = dp;
    d100 = dp+nComp;
    dp += xsize*nComp;
    d010 = dp;
    d110 = dp+nComp;
    dp += xsize*ysize*nComp;
    d011 = dp;
    d111 = dp+nComp;
    dp -= xsize*nComp;
    d001 = dp;
    d101 = dp+nComp;
    }
  else
    {
    d000 = INRANGE(x0, y0, z0) ? DENS(x0, y0, z0, moving_ptr, nComp) : def_value;
    d001 = INRANGE(x0, y0, z1) ? DENS(x0, y0, z1, moving_ptr, nComp) : def_value;
    d010 = INRANGE(x0, y1, z0) ? DENS(x0, y1, z0, moving_ptr, nComp) : def_value;
    d011 = INRANGE(x0, y1, z1) ? DENS(x0, y1, z1, moving_ptr, nComp) : def_value;
    d100 = INRANGE(x1, y0, z0) ? DENS(x1, y0, z0, moving_ptr, nComp) : def_value;
    d101 = INRANGE(x1, y0, z1) ? DENS(x1, y0, z1, moving_ptr, nComp) : def_value;
    d110 = INRANGE(x1, y1, z0) ? DENS(x1, y1, z0, moving_ptr, nComp) : def_value;
    d111 = INRANGE(x1, y1, z1) ? DENS(x1, y1, z1, moving_ptr, nComp) : def_value;
    }

  // Interpolate each component
  for(int iComp = 0; iComp < nComp; iComp+=stride)
    {
    // Interpolate first component
    dx00 = LERP(fx, *d000++, *d100++);
    dx01 = LERP(fx, *d001++, *d101++);
    dx10 = LERP(fx, *d010++, *d110++);
    dx11 = LERP(fx, *d011++, *d111++);
    dxy0 = LERP(fy, dx00, dx10);
    dxy1 = LERP(fy, dx01, dx11);
    *(out++) = LERP(fz, dxy0, dxy1);
    }
}

template <typename TImage>
class ImageRegionConstIteratorWithIndexOverride
    : public itk::ImageRegionConstIteratorWithIndex<TImage>
{
public:
  typedef ImageRegionConstIteratorWithIndexOverride<TImage> Self;
  typedef itk::ImageRegionConstIteratorWithIndex<TImage> Superclass;
  typedef typename Superclass::RegionType RegionType;
  typedef typename Superclass::InternalPixelType InternalPixelType;

  ImageRegionConstIteratorWithIndexOverride(TImage *im, const RegionType &region)
    : Superclass(im, region) {}

  const InternalPixelType *GetPosition() { return this->m_Position; }
  const InternalPixelType *GetBeginPosition() { return this->m_Begin; }
};


/**
 * Compute the output for the region specified by outputRegionForThread.
 */
template <class TInputImage, class TOutputImage, class TTransformTraits>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TTransformTraits>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  ThreadIdType threadId )
{
  // Get the pointers to the input and output images
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary"));
  InputImageType *moving = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("moving"));
  OutputImageType *out = this->GetOutput();

  // Get the number of components
  int kFixed = fixed->GetNumberOfComponentsPerPixel();
  int kMoving = moving->GetNumberOfComponentsPerPixel();

  // Iterate over the deformation field and the output image. In reality, we don't
  // need to waste so much time on iteration, so we use a specialized iterator here
  typedef ImageRegionConstIteratorWithIndexOverride<OutputImageType> OutputIter;

  // Location of the lookup
  vnl_vector_fixed<float, ImageDimension> cix;

  // Pointer to the fixed image data
  const typename InputImageType::InternalPixelType *bFix = fixed->GetBufferPointer();
  const typename InputImageType::InternalPixelType *bMov = moving->GetBufferPointer();
  typename OutputImageType::InternalPixelType *bOut = out->GetBufferPointer();

  // Pointer to store interpolated moving data
  vnl_vector<typename InputImageType::InternalPixelType> interp_mov(kMoving);
  SummaryType &sum_res = m_SummaryResultPerThread[threadId];

  // Get the stride for interpolation (how many moving pixels to skip)
  int stride = TTransformTraits::GetStride(kMoving);

  // Array of moving image size
  vnl_vector<int> mov_size(ImageDimension);
  for(unsigned int j = 0; j < ImageDimension; j++ )
    mov_size[j] = moving->GetBufferedRegion().GetSize()[j];

  // Array of zeros - default value
  vnl_vector<InputComponentType> zeros(kMoving, 0.0);

  // Iterate over the fixed space region
  for(OutputIter it(out, outputRegionForThread); !it.IsAtEnd(); ++it)
    {
    // Get the index at the current location
    const IndexType &idx = it.GetIndex();

    // Get the output pointer at this location
    typename OutputImageType::InternalPixelType *ptrOut =
        const_cast<typename OutputImageType::InternalPixelType *>(it.GetPosition());

    // Get the offset into the fixed pointer
    long offset = ptrOut - bOut;

    // Map to a position at which to interpolate
    TTransformTraits::TransformIndex(idx, m_Transform, offset, cix.data_block());

    // Perform the interpolation, put the results into interp_mov
    this->OpticalFlowFastInterpolate(
          Dispatch<ImageDimension>(),
          bMov, kMoving, stride, mov_size.data_block(), zeros.data_block(),
          cix.data_block(), interp_mov.data_block());

    // Perform the calculation of interest on the interpolated data
    TTransformTraits::PostInterpolate(
          idx, bFix + offset * kFixed,
          interp_mov.data_block(), kMoving, m_Weights.data_block(),
          sum_res.data_block(), *ptrOut);
    }

}




/**
 * Compute the output for the region specified by outputRegionForThread.
 */
/*
template <class TImage, class TVectorImage, class TFloat>
void
MultiImageOpticalFlowImageFilter<TImage,TVectorImage,TFloat>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  ThreadIdType threadId )
{
  // Get the pointers to the four images
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary"));
  InputImageType *moving = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("moving"));
  DeformationFieldType *def = dynamic_cast<DeformationFieldType *>(this->ProcessObject::GetInput("deformation"));
  OutputImageType *out = this->GetOutput();

  // Get the number of components
  int kFixed = fixed->GetNumberOfComponentsPerPixel();
  int kMoving = moving->GetNumberOfComponentsPerPixel();

  // Iterate over the deformation field and the output image. In reality, we don't
  // need to waste so much time on iteration, so we use a specialized iterator here
  typedef ImageRegionConstIteratorWithIndexOverride<OutputImageType> OutputIter;

  // Get the transform parameters if using affine transform
  vnl_matrix_fixed<float, ImageDimension, ImageDimension> t_M;
  vnl_vector_fixed<float, ImageDimension> t_b;
  if(m_Transform)
    {
    for(int i = 0; i < ImageDimension; i++)
      {
      for(int j = 0; j < ImageDimension; j++)
        {
        t_M(i,j) = m_Transform->GetMatrix().GetVnlMatrix()(i,j);
        }
      t_b(i) = m_Transform->GetOffset()(i);
      }
    }
  else
    {
    t_M.set_identity();
    t_b.fill(0.0f);
    }

  // Location of the lookup
  vnl_vector_fixed<float, ImageDimension> cix;

  // Pointer to the fixed image data
  const typename InputImageType::InternalPixelType *bFix = fixed->GetBufferPointer();
  const typename DeformationFieldType::InternalPixelType *bDef = def->GetBufferPointer();
  typename OutputImageType::InternalPixelType *bOut = out->GetBufferPointer();

  // Array of moving image size
  vnl_vector<int> mov_size(ImageDimension);
  for(unsigned int j = 0; j < ImageDimension; j++ )
    mov_size[j] = moving->GetBufferedRegion().GetSize()[j];

  // Array of zeros - default value
  vnl_vector<InputComponentType> zeros(kMoving, 0.0);

  // Iterate over the fixed space region
  for(OutputIter it(out, outputRegionForThread); !it.IsAtEnd(); ++it)
    {
    // Get the index at the current location
    const IndexType &idx = it.GetIndex();

    // Get the output pointer at this location
    const typename OutputImageType::InternalPixelType *ptrOut = it.GetPosition();

    // Get the offset into the fixed pointer
    long offset = ptrOut - bOut;

    // Map to a position at which to interpolate
    if(def)
      {
      const typename DeformationFieldType::InternalPixelType &def_x = bDef[offset];

      // Use deformation field
      for(unsigned int j = 0; j < ImageDimension; j++ )
        {
        cix[j] = idx[j] + m_DeformationScaling * def_x[j];
        }
      }
    else
      {
      // Compute the affine transform - directly in index space
      for(int i = 0; i < ImageDimension; i++)
        {
        cix[i] = t_b(i);
        for(int j = 0; j < ImageDimension; j++)
          cix[i] += t_M(i,j) * idx[j];
        }
      }

    // Call the interpolation code
    m_TotalEnergyPerThread[threadId] +=
        this->OpticalFlowFastInterpolate(
          Dispatch<ImageDimension>(),
          cix.data_block(),
          bFix + offset * kFixed,
          moving->GetBufferPointer(),
          *ptrOut,
          mov_size.data_block(),
          kMoving,
          zeros.data_block());
    }

}
*/

template <class TInputImage, class TOutputImage, class TTransformTraits>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TTransformTraits>
::GenerateInputRequestedRegion()
{
  // call the superclass's implementation
  Superclass::GenerateInputRequestedRegion();

  // Different behavior for fixed and moving images
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary"));
  InputImageType *moving = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("moving"));
  ImageBase<ImageDimension> *transform = TTransformTraits::AsImageBase(m_Transform);
  // DeformationFieldType *def = dynamic_cast<DeformationFieldType *>(this->ProcessObject::GetInput("deformation"));

  if(moving)
    moving->SetRequestedRegionToLargestPossibleRegion();

  if(fixed)
    {
    fixed->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );
    if(!fixed->VerifyRequestedRegion())
      fixed->SetRequestedRegionToLargestPossibleRegion();
    }

  if(transform)
    {
    transform->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );
    if(!transform->VerifyRequestedRegion())
      transform->SetRequestedRegionToLargestPossibleRegion();
    }
}


template <class TInputImage, class TOutputImage, class TTransformTraits>
void
MultiImageOpticalFlowImageFilter<TInputImage,TOutputImage,TTransformTraits>
::GenerateOutputInformation()
{
  // call the superclass's implementation of this method
  Superclass::GenerateOutputInformation();

  OutputImageType *outputPtr = this->GetOutput();
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary"));
  outputPtr->SetSpacing( fixed->GetSpacing() );
  outputPtr->SetOrigin( fixed->GetOrigin() );
  outputPtr->SetDirection( fixed->GetDirection() );
  outputPtr->SetLargestPossibleRegion( fixed->GetLargestPossibleRegion() );
}


} // end namespace itk

#endif
