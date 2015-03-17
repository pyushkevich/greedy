/*=========================================================================

  Program:   ALFABIS fast image registration
  Language:  C++

  Copyright (c) Paul Yushkevich. All rights reserved.

  This program is part of ALFABIS

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
#ifndef __MultiComponentNCCImageMetric_txx
#define __MultiComponentNCCImageMetric_txx

#include "MultiComponentNCCImageMetric.h"
#include "OneDimensionalInPlaceAccumulateFilter.h"





/** \class MultiImageNCCPrecomputeFilter
 * \brief Warps an image using an input deformation field (for LDDMM)
 *
 * This filter takes a pair of images plus a warp and computes the components that
 * are used to calculate the cross-correlation metric between them and
 * the gradient. These components are in the form I, I*J, I * gradJ, and
 * so on. These components must then be mean-filtered and combined to get the
 * metric and the gradient.
 *
 * The output of this filter must be a vector image. The input may be a vector image.
 *
 */
template <class TMetricTraits, class TOutputImage>
class ITK_EXPORT MultiImageNCCPrecomputeFilter :
    public itk::ImageToImageFilter<typename TMetricTraits::InputImageType, TOutputImage>
{
public:

  /** Types from the traits */
  typedef typename TMetricTraits::InputImageType        InputImageType;
  typedef typename TMetricTraits::MaskImageType         MaskImageType;
  typedef typename TMetricTraits::DeformationFieldType  DeformationFieldType;
  typedef typename TMetricTraits::MetricImageType       MetricImageType;
  typedef typename TMetricTraits::GradientImageType     GradientImageType;
  typedef TOutputImage                                  OutputImageType;


  /** Standard class typedefs. */
  typedef MultiImageNCCPrecomputeFilter                         Self;
  typedef itk::ImageToImageFilter<InputImageType, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                               Pointer;
  typedef itk::SmartPointer<const Self>                         ConstPointer;

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef typename InputImageType::PixelType          InputPixelType;
  typedef typename InputImageType::InternalPixelType  InputComponentType;
  typedef typename TOutputImage::PixelType            OutputPixelType;
  typedef typename TOutputImage::InternalPixelType    OutputComponentType;
  typedef typename DeformationFieldType::PixelType    DeformationVectorType;
  typedef typename MetricImageType::PixelType         MetricPixelType;
  typedef typename GradientImageType::PixelType       GradientPixelType;
  typedef typename MetricImageType::IndexType         IndexType;
  typedef typename MetricImageType::IndexValueType    IndexValueType;
  typedef typename MetricImageType::SizeType          SizeType;
  typedef typename MetricImageType::SpacingType       SpacingType;
  typedef typename MetricImageType::DirectionType     DirectionType;

  typedef typename Superclass::DataObjectIdentifierType DataObjectIdentifierType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageNCCPrecomputeFilter, ImageToImageFilter )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension );

  itkBooleanMacro(ComputeGradient)

  /**
   * Whether the gradient of the NCC metric will be computed. Depending on this, the filter
   * will generate 5 components per pixel (x,y,xy,x2,y2) or a bunch more needed for the
   * gradient computation. Default is Off.
   */
  void SetComputeGradient(bool flag)
  {
    this->m_ComputeGradient = flag;
    this->UpdateOutputs();
  }

  itkBooleanMacro(ComputeMovingDomainMask)

  /** Specify whether the filter should compute a mask based on the moving image domain */
  void SetComputeMovingDomainMask(bool flag)
  {
    this->m_ComputeMovingDomainMask = flag;
    this->UpdateOutputs();
  }


  /** Set the fixed image(s) */
  void SetFixedImage(InputImageType *fixed)
    { this->itk::ProcessObject::SetInput("Primary", fixed); }

  InputImageType *GetFixedImage()
    { return dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("Primary")); }

  /** Set the moving image(s) and their gradients */
  void SetMovingImage(InputImageType *moving)
    { this->itk::ProcessObject::SetInput("moving", moving); }

  InputImageType *GetMovingImage()
    { return dynamic_cast<InputImageType *>(this->ProcessObject::GetInput("moving")); }

  /**
   * Set the deformation field. An affine transformation should be converted
   * to a deformation field first
   */
  void SetDeformationField(DeformationFieldType *phi)
    { this->itk::ProcessObject::SetInput("phi", phi); }
\
  DeformationFieldType *GetDeformationField()
    { return dynamic_cast<DeformationFieldType *>(this->ProcessObject::GetInput("phi")); }

  /** Get the metric output */
  MetricImageType *GetMetricOutput()
    { return dynamic_cast<MetricImageType *>(this->ProcessObject::GetOutput("Primary")); }

  /** Get the mask output */
  MetricImageType *GetMovingDomainMaskOutput()
    { return dynamic_cast<MetricImageType *>(this->ProcessObject::GetOutput("moving_mask")); }

  /** Get the metric output */
  GradientImageType *GetGradientOutput()
    { return dynamic_cast<GradientImageType *>(this->ProcessObject::GetOutput("gradient")); }

  /** Get the mask output */
  GradientImageType *GetMovingDomainMaskGradientOutput()
    { return dynamic_cast<GradientImageType *>(this->ProcessObject::GetOutput("moving_mask_gradient")); }


  /** Get the number of components in the output */
  int GetNumberOfOutputComponents();


protected:
  MultiImageNCCPrecomputeFilter();
  ~MultiImageNCCPrecomputeFilter() {}

  /** SimpleWarpImageFilter is implemented as a multi-threaded filter.
   * As such, it needs to provide and implementation for
   * ThreadedGenerateData(). */
  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            itk::ThreadIdType threadId );

  /** Set up the output information */
  virtual void GenerateOutputInformation();

  /** Override input checks to allow fixed and moving to be in different space */
  virtual void VerifyInputInformation() {}



  virtual typename itk::DataObject::Pointer MakeOutput(const DataObjectIdentifierType &);

  void UpdateOutputs();
  void ToggleOutput(bool flag, const DataObjectIdentifierType &key);

private:
  MultiImageNCCPrecomputeFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // Whether the gradient of the NCC metric will be computed. Depending on this, the filter
  // will generate 5 components per pixel (x,y,xy,x2,y2) or a bunch more needed for the
  // gradient computation.
  bool m_ComputeGradient;
  bool m_ComputeMovingDomainMask;
};





template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
class MultiImageNCCPostcomputeFilter : public itk::ImageToImageFilter<TInputImage, TMetricImage>
{
public:

  /** Standard class typedefs. */
  typedef MultiImageNCCPostcomputeFilter                      Self;
  typedef itk::ImageToImageFilter<TInputImage,TMetricImage>   Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods) */
  itkTypeMacro( MultiImageNCCPostcomputeFilter, ImageToImageFilter )

  /** Determine the image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension );

  /** Typedef to describe the output image region type. */
  typedef typename Superclass::OutputImageRegionType         OutputImageRegionType;

  /** Inherit some types from the superclass. */
  typedef TInputImage                                 InputImageType;
  typedef typename TInputImage::PixelType             InputPixelType;
  typedef typename TInputImage::InternalPixelType     InputComponentType;
  typedef TMetricImage                                MetricImageType;
  typedef typename MetricImageType::PixelType         MetricPixelType;
  typedef TGradientImage                              GradientImageType;
  typedef typename GradientImageType::PixelType       GradientPixelType;
  typedef TMaskImage                                  MaskImageType;
  typedef typename MaskImageType::PixelType           MaskPixelType;
  typedef typename InputImageType::IndexType          IndexType;
  typedef typename InputImageType::SizeType           SizeType;

  typedef typename Superclass::DataObjectPointerArraySizeType  DataObjectPointerArraySizeType;


  /** Weight vector */
  typedef vnl_vector<float>                           WeightVectorType;

  /** Is the gradient of the metric being computed */
  itkSetMacro(ComputeGradient, bool)
  itkBooleanMacro(ComputeGradient)

  /** Set the weight vector */
  itkSetMacro(Weights, WeightVectorType)
  itkGetConstMacro(Weights, WeightVectorType)

  /** Set the mask image */
  void SetMaskImage(MaskImageType *mask)
    { this->itk::ProcessObject::SetInput("mask", mask); }

  /** Get the metric image */
  MetricImageType *GetMetricOutput();

  /** Get the gradient image */
  GradientImageType *GetGradientOutput();

  /** Get the metric value */
  itkGetMacro(MetricValue, double)

protected:

  MultiImageNCCPostcomputeFilter();
  ~MultiImageNCCPostcomputeFilter() {}

  virtual void BeforeThreadedGenerateData();
  virtual void AfterThreadedGenerateData();

  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            itk::ThreadIdType threadId );

  typename itk::DataObject::Pointer MakeOutput(DataObjectPointerArraySizeType idx);

  // Whether or not the gradient is computed
  bool m_ComputeGradient;

  // Weight vector
  WeightVectorType m_Weights;

  // Vector of accumulated data (difference, gradient of affine transform, etc)
  double                          m_MetricValue;
  std::vector<double>             m_MetricPerThread;

  // TODO: handle this better!
  typename MetricImageType::Pointer m_MetricImage;
};








template <class TMetricTraits, class TOutputImage>
MultiImageNCCPrecomputeFilter<TMetricTraits,TOutputImage>
::MultiImageNCCPrecomputeFilter()
{
  m_ComputeGradient = false;
  m_ComputeMovingDomainMask = false;

  // Create the outputs of this filter
  this->SetPrimaryOutput(this->MakeOutput("Primary"));
}

template <class TMetricTraits, class TOutputImage>
typename itk::DataObject::Pointer
MultiImageNCCPrecomputeFilter<TMetricTraits,TOutputImage>
::MakeOutput(const DataObjectIdentifierType &key)
{
  if(key == "Primary")
    {
    return (OutputImageType::New()).GetPointer();
    }
  if(key == "moving_mask")
    {
    return (MetricImageType::New()).GetPointer();
    }
  else if(key == "moving_mask_gradient")
    {
    return (GradientImageType::New()).GetPointer();
    }
  else
    {
    return NULL;
    }
}

template <class TMetricTraits, class TOutputImage>
void
MultiImageNCCPrecomputeFilter<TMetricTraits,TOutputImage>
::ToggleOutput(bool flag, const DataObjectIdentifierType &key)
{
  if(flag && !this->HasOutput(key))
    this->SetOutput(key, this->MakeOutput(key));
  if(!flag && this->HasOutput(key))
    this->RemoveOutput(key);
}

template <class TMetricTraits, class TOutputImage>
void
MultiImageNCCPrecomputeFilter<TMetricTraits,TOutputImage>
::UpdateOutputs()
{
  this->ToggleOutput(m_ComputeMovingDomainMask, "moving_mask");
  this->ToggleOutput(m_ComputeGradient && m_ComputeMovingDomainMask, "moving_mask_gradient");
}


/**
 * Generate output information, which will be different from the default
 */
template <class TMetricTraits, class TOutputImage>
void
MultiImageNCCPrecomputeFilter<TMetricTraits,TOutputImage>
::GenerateOutputInformation()
{
  // Call the parent method to set up all the outputs
  Superclass::GenerateOutputInformation();

  // Set the number of components in the primary output
  int ncomp = this->GetNumberOfOutputComponents();
  this->GetOutput()->SetNumberOfComponentsPerPixel(ncomp);
}

template <class TMetricTraits, class TOutputImage>
int
MultiImageNCCPrecomputeFilter<TMetricTraits,TOutputImage>
::GetNumberOfOutputComponents()
{
  int ncomp = (m_ComputeGradient)
              ? 1 + this->GetInput()->GetNumberOfComponentsPerPixel() * (5 + ImageDimension * 3)
              : 1 + this->GetInput()->GetNumberOfComponentsPerPixel() * 5;
  return ncomp;
}

// #define _FAKE_FUNC_

/**
 * Compute the output for the region specified by outputRegionForThread.
 */
template <class TMetricTraits, class TOutputImage>
void
MultiImageNCCPrecomputeFilter<TMetricTraits,TOutputImage>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  itk::ThreadIdType threadId )
{
  // Get the pointers to the input and output images
  InputImageType *fixed = dynamic_cast<InputImageType *>(this->itk::ProcessObject::GetInput("Primary"));
  InputImageType *moving = dynamic_cast<InputImageType *>(this->itk::ProcessObject::GetInput("moving"));
  DeformationFieldType *phi = dynamic_cast<DeformationFieldType *>(this->itk::ProcessObject::GetInput("phi"));
  OutputImageType *out = this->GetOutput();

  // Get the number of components
  int kFixed = fixed->GetNumberOfComponentsPerPixel();
  int kOutput = out->GetNumberOfComponentsPerPixel();

  // Iterate over the deformation field and the output image. In reality, we don't
  // need to waste so much time on iteration, so we use a specialized iterator here
  typedef itk::ImageLinearConstIteratorWithIndex<OutputImageType> OutputIterBase;
  typedef IteratorExtender<OutputIterBase> OutputIter;

  // Location of the lookup
  vnl_vector_fixed<float, ImageDimension> cix;

  // Pointer to the fixed image data
  const InputComponentType *bFix = fixed->GetBufferPointer();
  const DeformationVectorType *bPhi = phi->GetBufferPointer();
  OutputComponentType *bOut = out->GetBufferPointer();

  // Pointers to mask output data  GradientPixelType *bGradient = (this->m_ComputeGradient)
  MetricPixelType *bMDMask = (this->m_ComputeMovingDomainMask)
                             ? this->GetMovingDomainMaskOutput()->GetBufferPointer()
                             : NULL;

  GradientPixelType *bMDMaskGradient = (this->m_ComputeGradient && this->m_ComputeMovingDomainMask)
                                       ? this->GetMovingDomainMaskGradientOutput()->GetBufferPointer()
                                       : NULL;

  // Pointer to store interpolated moving data
  vnl_vector<InputComponentType> interp_mov(kFixed);

  // Pointer to store the gradient of the moving images
  vnl_vector<InputComponentType> interp_mov_grad(kFixed * ImageDimension);

  // Create an interpolator for the moving image
  typedef FastLinearInterpolator<InputComponentType, ImageDimension> FastInterpolator;
  FastInterpolator flint(moving);

  // Iterate over the fixed space region
  for(OutputIter it(out, outputRegionForThread); !it.IsAtEnd(); it.NextLine())
    {
    // Process the whole line using pointer arithmetic. We have to deal with messy behavior
    // of iterators on vector images. Trying to avoid using accessors and Set/Get
    long offset_in_pixels = it.GetPosition() - bOut;
    OutputComponentType *ptrOut = bOut + offset_in_pixels * kOutput;
    OutputComponentType *ptrEnd = ptrOut + outputRegionForThread.GetSize(0) * kOutput;

    // Get the beginning of the same line in the deformation image
    const DeformationVectorType *def_ptr = bPhi + offset_in_pixels;

    // Get the beginning of the same line in the fixed image
    const InputComponentType *fix_ptr = bFix + offset_in_pixels * kFixed;

    // Get the index at the current location
    IndexType idx = it.GetIndex();

    // Get the output pointer for the mask and mask gradient
    MetricPixelType *ptrMDMask = (m_ComputeMovingDomainMask) ? bMDMask + offset_in_pixels : NULL;
    GradientPixelType *ptrMDMaskGradient =
        (m_ComputeMovingDomainMask && m_ComputeGradient)
        ? bMDMaskGradient + offset_in_pixels : NULL;

    // Loop over the line
    for(; ptrOut < ptrEnd; idx[0]++, def_ptr++)
      {
      // Pointer to the data storing the interpolated moving values
      InputComponentType *mov_ptr = interp_mov.data_block();
      const InputComponentType *mov_ptr_end = mov_ptr + kFixed;

      // Where the gradient is placed
      InputComponentType *mov_grad_ptr = interp_mov_grad.data_block();

      // Map to a position at which to interpolate
      for(int i = 0; i < ImageDimension; i++)
        cix[i] = idx[i] + (*def_ptr)[i];

      typename FastInterpolator::InOut status;


#ifdef _FAKE_FUNC_

      // Fake a function
      double x = cix[0], y = cix[1], z = cix[2];
      double a = 0.01, b = 0.005, c = 0.008, d = 0.004;
      interp_mov[0] = sin(a * x * y + b * z) + cos(c * x + d * y * z);
      interp_mov_grad[0] = cos(a * x * y + b * z) * a * y - sin(c * x + d * y * z) * c;
      interp_mov_grad[1] = cos(a * x * y + b * z) * a * x - sin(c * x + d * y * z) * d * z;
      interp_mov_grad[2] = cos(a * x * y + b * z) * b     - sin(c * x + d * y * z) * d * y;
      status = FastInterpolator::INSIDE;

#else

      if(m_ComputeGradient)
        {
        // Compute gradient
        status = flint.InterpolateWithGradient(cix.data_block(), mov_ptr, mov_grad_ptr);
        }
      else
        {
        // Just interpolate
        status = flint.Interpolate(cix.data_block(), mov_ptr);
        }

#endif

      // Handle outside values
      if(status == FastInterpolator::OUTSIDE)
        {
        interp_mov.fill(0.0);
        if(m_ComputeGradient)
          interp_mov_grad.fill(0.0);
        }

      // Write out 1!
      *ptrOut++ = 1.0;

      for( ;mov_ptr < mov_ptr_end; ++mov_ptr, ++fix_ptr)
        {
        InputComponentType x_fix = *fix_ptr, x_mov = *mov_ptr;
        *ptrOut++ = x_fix;
        *ptrOut++ = x_mov;
        *ptrOut++ = x_fix * x_fix;
        *ptrOut++ = x_mov * x_mov;
        *ptrOut++ = x_fix * x_mov;

        if(m_ComputeGradient)
          {
          for(int i = 0; i < ImageDimension; i++, mov_grad_ptr++)
            {
            InputComponentType x_grad_mov_i = *mov_grad_ptr;
            *ptrOut++ = x_grad_mov_i;
            *ptrOut++ = x_fix * x_grad_mov_i;
            *ptrOut++ = x_mov * x_grad_mov_i;
            }
          }
        }

      // Handle the mask information
      if(m_ComputeMovingDomainMask)
        {
        if(status == FastInterpolator::BORDER)
          {
          if(m_ComputeGradient)
            {
            *ptrMDMask++ = flint.GetMaskAndGradient((*ptrMDMaskGradient++).GetDataPointer());
            }
          else
            {
            *ptrMDMask++ = flint.GetMask();
            }
          }
        else
          {
          *ptrMDMask++ = (status == FastInterpolator::INSIDE) ? 1.0 : 0.0;
          if(m_ComputeGradient)
            (*ptrMDMaskGradient++).Fill(0.0);
          }
        }
      }
    }
}

template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::MultiImageNCCPostcomputeFilter()
{
  // Set the number of outputs
  this->SetNumberOfRequiredOutputs(2);
  this->SetNthOutput(0, this->MakeOutput(0));
  this->SetNthOutput(1, this->MakeOutput(1));

  // We are not computing the gradient by default
  m_ComputeGradient = false;
}

template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
typename itk::DataObject::Pointer
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::MakeOutput(DataObjectPointerArraySizeType idx)
{
  if(idx == 0)
    return (MetricImageType::New()).GetPointer();
  else if(idx == 1)
    return (GradientImageType::New()).GetPointer();
  else
    return NULL;
}

template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
typename MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>::MetricImageType *
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::GetMetricOutput()
{
  return dynamic_cast<MetricImageType *>(this->ProcessObject::GetOutput(0));
}

template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
typename MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>::GradientImageType *
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::GetGradientOutput()
{
  return dynamic_cast<GradientImageType *>(this->ProcessObject::GetOutput(1));
}

template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
void
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::BeforeThreadedGenerateData()
{
  // Create the prototype results vector
  m_MetricPerThread.resize(this->GetNumberOfThreads(), 0.0);
}

/**
 * Setup state of filter after multi-threading.
 */
template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
void
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::AfterThreadedGenerateData()
{
  m_MetricValue = 0.0;
  for(int i = 0; i < m_MetricPerThread.size(); i++)
    m_MetricValue += m_MetricPerThread[i];
}

// Compute sigma_I, sigma_J, sigma_IJ
/*
 * COMPUTATION WITH EPSILON IN DENOM
 *
InputComponentType x_fix = *ptr++;
InputComponentType x_mov = *ptr++;
InputComponentType x_fix_sq = *ptr++;
InputComponentType x_mov_sq = *ptr++;
InputComponentType x_fix_mov = *ptr++;

InputComponentType x_fix_over_n = x_fix * one_over_n;
InputComponentType x_mov_over_n = x_mov * one_over_n;

InputComponentType var_fix = x_fix_sq - x_fix * x_fix_over_n;
InputComponentType var_mov = x_mov_sq - x_mov * x_mov_over_n;
InputComponentType cov_fix_mov = x_fix_mov - x_fix * x_mov_over_n;

InputComponentType one_over_denom = 1.0 / (var_fix * var_mov + eps);
InputComponentType cov_fix_mov_over_denom = cov_fix_mov * one_over_denom;
InputComponentType ncc_fix_mov = cov_fix_mov * cov_fix_mov_over_denom;

for(int i = 0; i < ImageDimension; i++)
  {
  InputComponentType x_grad_mov_i = *ptr++;
  InputComponentType x_fix_grad_mov_i = *ptr++;
  InputComponentType x_mov_grad_mov_i = *ptr++;

  // Derivative of cov_fix_mov
  InputComponentType grad_cov_fix_mov_i = x_fix_grad_mov_i - x_fix_over_n * x_grad_mov_i;

  // One half derivative of var_mov
  InputComponentType half_grad_var_mov_i = x_mov_grad_mov_i - x_mov_over_n * x_grad_mov_i;

  InputComponentType grad_ncc_fix_mov_i =
      2 * cov_fix_mov_over_denom * (grad_cov_fix_mov_i - var_fix * half_grad_var_mov_i * cov_fix_mov_over_denom);

  (*ptr_gradient)[i] += m_Weights[i_wgt] * grad_ncc_fix_mov_i;
  // (*ptr_gradient)[i] = grad_ncc_fix_mov_i;


  // (*ptr_gradient)[i] = x_grad_mov_i; // grad_cov_fix_mov_i;
  }

// *ptr_metric = ncc_fix_mov;
*ptr_metric += m_Weights[i_wgt] * ncc_fix_mov;
// *ptr_metric = x_mov; // cov_fix_mov;

++i_wgt;
*/


/*
 * ADD epsilon to numerator and denominator
 *

InputComponentType x_fix = *ptr++;
InputComponentType x_mov = *ptr++;
InputComponentType x_fix_sq = *ptr++;
InputComponentType x_mov_sq = *ptr++;
InputComponentType x_fix_mov = *ptr++;

InputComponentType x_fix_over_n = x_fix * one_over_n;
InputComponentType x_mov_over_n = x_mov * one_over_n;

// Epsilon is used to stabilize numerical computation
double eps = 1.0e-4;

InputComponentType var_fix = x_fix_sq - x_fix * x_fix_over_n + eps;
InputComponentType var_mov = x_mov_sq - x_mov * x_mov_over_n + eps;

InputComponentType cov_fix_mov = x_fix_mov - x_fix * x_mov_over_n + eps;

InputComponentType one_over_denom = 1.0 / (var_fix * var_mov);
InputComponentType cov_fix_mov_over_denom = cov_fix_mov * one_over_denom;
InputComponentType ncc_fix_mov = cov_fix_mov * cov_fix_mov_over_denom;

float w = m_Weights[i_wgt];
if(cov_fix_mov < 0)
  w = -w;

for(int i = 0; i < ImageDimension; i++)
  {
  InputComponentType x_grad_mov_i = *ptr++;
  InputComponentType x_fix_grad_mov_i = *ptr++;
  InputComponentType x_mov_grad_mov_i = *ptr++;

  // Derivative of cov_fix_mov
  InputComponentType grad_cov_fix_mov_i = x_fix_grad_mov_i - x_fix_over_n * x_grad_mov_i;

  // One half derivative of var_mov
  InputComponentType half_grad_var_mov_i = x_mov_grad_mov_i - x_mov_over_n * x_grad_mov_i;

  InputComponentType grad_ncc_fix_mov_i =
      2 * cov_fix_mov_over_denom * (grad_cov_fix_mov_i - var_fix * half_grad_var_mov_i * cov_fix_mov_over_denom);

  (*ptr_gradient)[i] += w * grad_ncc_fix_mov_i;
  // (*ptr_gradient)[i] = grad_ncc_fix_mov_i;


  // (*ptr_gradient)[i] = x_grad_mov_i; // grad_cov_fix_mov_i;
  }

// *ptr_metric = ncc_fix_mov;

*ptr_metric += w * ncc_fix_mov;
// *ptr_metric = x_mov; // cov_fix_mov;
*/


template <class TPixel, class TWeight, class TMetric, class TGradient>
TPixel *
MultiImageNNCPostComputeFunction(
    TPixel *ptr, TPixel *ptr_end, TWeight *weights, TMetric *ptr_metric, TGradient *ptr_gradient, int ImageDimension)
{
  // Get the size of the mean filter kernel
  TPixel n = *ptr++, one_over_n = 1.0 / n;

  // Loop over components
  int i_wgt = 0;
  const TPixel eps = 1e-8;

  // Initialize metric to zero
  *ptr_metric = 0;

  for(; ptr < ptr_end; ++i_wgt)
    {
    TPixel x_fix = *ptr++;
    TPixel x_mov = *ptr++;
    TPixel x_fix_sq = *ptr++;
    TPixel x_mov_sq = *ptr++;
    TPixel x_fix_mov = *ptr++;

    TPixel x_fix_over_n = x_fix * one_over_n;
    TPixel x_mov_over_n = x_mov * one_over_n;

    TPixel var_fix = x_fix_sq - x_fix * x_fix_over_n;
    TPixel var_mov = x_mov_sq - x_mov * x_mov_over_n;

    if(var_fix < eps || var_mov < eps)
      {
      if(ptr_gradient)
        ptr += 3 * ImageDimension;
      continue;
      }

    TPixel cov_fix_mov = x_fix_mov - x_fix * x_mov_over_n;
    TPixel one_over_denom = 1.0 / (var_fix * var_mov);
    TPixel cov_fix_mov_over_denom = cov_fix_mov * one_over_denom;
    TPixel ncc_fix_mov = cov_fix_mov * cov_fix_mov_over_denom;

    // Weight - includes scaling of squared covariance by direction
    TWeight w = (cov_fix_mov < 0) ? -weights[i_wgt] : weights[i_wgt];

    if(ptr_gradient)
      {
      for(int i = 0; i < ImageDimension; i++)
        {
        TPixel x_grad_mov_i = *ptr++;
        TPixel x_fix_grad_mov_i = *ptr++;
        TPixel x_mov_grad_mov_i = *ptr++;

        // Derivative of cov_fix_mov
        TPixel grad_cov_fix_mov_i = x_fix_grad_mov_i - x_fix_over_n * x_grad_mov_i;

        // One half derivative of var_mov
        TPixel half_grad_var_mov_i = x_mov_grad_mov_i - x_mov_over_n * x_grad_mov_i;

        TPixel grad_ncc_fix_mov_i =
            2 * cov_fix_mov_over_denom * (grad_cov_fix_mov_i - var_fix * half_grad_var_mov_i * cov_fix_mov_over_denom);

        (*ptr_gradient)[i] += w * grad_ncc_fix_mov_i;
        }
      }

    // Accumulate the metric
    *ptr_metric += w * ncc_fix_mov;
    }

  return ptr;
}


template <class TInputImage, class TMetricImage, class TGradientImage, class TMaskImage>
void
MultiImageNCCPostcomputeFilter<TInputImage,TMetricImage,TGradientImage,TMaskImage>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  itk::ThreadIdType threadId )
{
  // Set up the iterators for the three images. In the future, check if the
  // iteration contributes in any way to the filter cost, and consider more
  // direct, faster approaches
  typedef itk::ImageLinearConstIteratorWithIndex<InputImageType> InputIteratorTypeBase;
  typedef IteratorExtender<InputIteratorTypeBase> InputIteratorType;
  typedef itk::ImageRegionIterator<TMetricImage> MetricIteratorType;
  typedef itk::ImageRegionIterator<TGradientImage> GradientIteratorType;

  InputImageType *image = const_cast<InputImageType *>(this->GetInput());
  InputIteratorType it_input(image, outputRegionForThread);

  // Get the mask image (optional)
  const MaskImageType *mask = dynamic_cast<MaskImageType *>(this->itk::ProcessObject::GetInput("mask"));

  // Number of input components
  int nc = this->GetInput()->GetNumberOfComponentsPerPixel();

  // Accumulated metric
  m_MetricPerThread[threadId] = 0.0;

  // Iterate over lines for greater efficiency
  for(; !it_input.IsAtEnd(); it_input.NextLine())
    {
    // Get the pointer to the start of the current line
    long offset_in_pixels = it_input.GetPosition() - image->GetBufferPointer();
    const InputComponentType *line_begin = image->GetBufferPointer() + offset_in_pixels * nc,
        *ptr = line_begin;
    const InputComponentType *line_end = ptr + outputRegionForThread.GetSize(0) * nc;

    // Get the offset into the metric and gradient images
    MetricPixelType *ptr_metric = this->GetMetricOutput()->GetBufferPointer() + offset_in_pixels;

    // The gradient output is optional
    GradientPixelType *ptr_gradient = (this->m_ComputeGradient)
                                      ? this->GetGradientOutput()->GetBufferPointer() + offset_in_pixels
                                      : NULL;

    // Four versions of the code - depending on the mask and gradient
    if(mask)
      {
      if(ptr_gradient)
        {
        // Get the offset into the mask, if a mask exists
        const MaskPixelType *ptr_mask = mask->GetBufferPointer() + offset_in_pixels;

        for(; ptr < line_end; ++ptr_metric, ++ptr_gradient, ++ptr_mask)
          {
          *ptr_metric = itk::NumericTraits<MetricPixelType>::Zero;
          *ptr_gradient = itk::NumericTraits<GradientPixelType>::Zero;

          // Should we skip this pixel?
          MaskPixelType mask_val = *ptr_mask;
          if(mask_val == 0)
            {
            ptr += nc;
            }
          else
            {
            // End of the chunk for this pixel
            const InputComponentType *ptr_end = ptr + nc;

            // Apply the post computation
            ptr = MultiImageNNCPostComputeFunction(ptr, ptr_end, m_Weights.data_block(), ptr_metric, ptr_gradient, ImageDimension);

            // Scale metric and gradient by the mask
            *ptr_metric *= mask_val;
            *ptr_gradient *= mask_val;

            // Accumulate the summary metric
            m_MetricPerThread[threadId] += *ptr_metric;
            }
          }
        }
      else
        {
        // Get the offset into the mask, if a mask exists
        const MaskPixelType *ptr_mask = mask->GetBufferPointer() + offset_in_pixels;

        for(; ptr < line_end; ++ptr_metric, ++ptr_mask)
          {
          *ptr_metric = itk::NumericTraits<MetricPixelType>::Zero;

          // Should we skip this pixel?
          MaskPixelType mask_val = *ptr_mask;
          if(mask_val == 0)
            {
            ptr += nc;
            }
          else
            {
            // End of the chunk for this pixel
            const InputComponentType *ptr_end = ptr + nc;

            // Apply the post computation
            ptr = MultiImageNNCPostComputeFunction(ptr, ptr_end, m_Weights.data_block(), ptr_metric, ptr_gradient, ImageDimension);

            // Scale metric and gradient by the mask
            *ptr_metric *= mask_val;

            // Accumulate the summary metric
            m_MetricPerThread[threadId] += *ptr_metric;
            }
          }
        }
      }
    else
      {
      if(ptr_gradient)
        {
        for(; ptr < line_end; ++ptr_metric, ++ptr_gradient)
          {
          *ptr_metric = itk::NumericTraits<MetricPixelType>::Zero;
          *ptr_gradient = itk::NumericTraits<GradientPixelType>::Zero;

          // End of the chunk for this pixel
          const InputComponentType *ptr_end = ptr + nc;

          // Apply the post computation
          ptr = MultiImageNNCPostComputeFunction(ptr, ptr_end, m_Weights.data_block(), ptr_metric, ptr_gradient, ImageDimension);

          // Accumulate the summary metric
          m_MetricPerThread[threadId] += *ptr_metric;
          }
        }
      else
        {
        for(; ptr < line_end; ++ptr_metric)
          {
          *ptr_metric = itk::NumericTraits<MetricPixelType>::Zero;

          // End of the chunk for this pixel
          const InputComponentType *ptr_end = ptr + nc;

          // Apply the post computation
          ptr = MultiImageNNCPostComputeFunction(ptr, ptr_end, m_Weights.data_block(), ptr_metric, ptr_gradient, ImageDimension);

          // Accumulate the summary metric
          m_MetricPerThread[threadId] += *ptr_metric;
          }
        }
      }
    }
}




template <class TMetricTraits>
void
MultiComponentNCCImageMetric<TMetricTraits>
::GenerateData()
{
  // Create the mini-pipeline of filters

  // Pre-compute filter
  typedef MultiImageNCCPrecomputeFilter<TMetricTraits, InputImageType> PreFilterType;
  typename PreFilterType::Pointer preFilter = PreFilterType::New();

  // Configure the precompute filter
  preFilter->SetComputeGradient(this->m_ComputeGradient);
  preFilter->SetComputeMovingDomainMask(this->m_ComputeMovingDomainMask);
  preFilter->SetFixedImage(this->GetFixedImage());
  preFilter->SetMovingImage(this->GetMovingImage());
  preFilter->SetDeformationField(this->GetDeformationField());

  // Number of components in the working image
  int ncomp = preFilter->GetNumberOfOutputComponents();

  // If the user supplied a working image, configure it and graft it as output
  if(!m_WorkingImage)
    {
    // Configure the working image
    m_WorkingImage->CopyInformation(this->GetFixedImage());
    m_WorkingImage->SetNumberOfComponentsPerPixel(ncomp);
    m_WorkingImage->SetRegions(this->GetFixedImage()->GetBufferedRegion());
    m_WorkingImage->Allocate();

    // Graft the working image onto the filter's output
    // TODO: preFilter->GraftOutput(m_WorkingImage);
    }

  // If the filter needs moving domain mask/gradient, graft those as well
  if(this->m_ComputeMovingDomainMask)
    {
    preFilter->GetMovingDomainMaskOutput()->Graft(this->GetMovingDomainMaskOutput());
    if(this->m_ComputeGradient)
      preFilter->GetMovingDomainMaskGradientOutput()->Graft(
            this->GetMovingDomainMaskGradientOutput());
    }

  // Execute the filter
  preFilter->Update();

#ifdef DUMP_NCC
  typename itk::ImageFileWriter<MultiComponentImageType>::Pointer pwriter = itk::ImageFileWriter<MultiComponentImageType>::New();
  pwriter->SetInput(preFilter->GetOutput());
  pwriter->SetFileName("nccpre.nii.gz");
  pwriter->Update();
#endif

  // Currently, we have all the stuff we need to compute the metric in the working
  // image. Next, we run the fast sum computation to give us the local average of
  // intensities, products, gradients in the working image
  typedef OneDimensionalInPlaceAccumulateFilter<InputImageType> AccumFilterType;

  // Create a chain of separable 1-D filters
  typename itk::ImageSource<InputImageType>::Pointer pipeTail;
  for(int dir = 0; dir < ImageDimension; dir++)
    {
    typename AccumFilterType::Pointer accum = AccumFilterType::New();
    if(pipeTail.IsNull())
      accum->SetInput(preFilter->GetOutput());
    else
      accum->SetInput(pipeTail->GetOutput());
    accum->SetDimension(dir);
    accum->SetRadius(m_Radius[dir]);
    pipeTail = accum;

    accum->Update();
    }

#ifdef DUMP_NCC
  pwriter->SetInput(pipeTail->GetOutput());
  pwriter->SetFileName("nccaccum.nii.gz");
  pwriter->Update();
#endif

  // Now pipetail has the mean filtering of the different components in m_NCCWorkingImage.
  // Last piece is to perform a calculation that will convert all this information into a
  // metric value and a gradient value. For the time being, we will use the unary functor
  // image filter to compute this, but a slightly more efficient implementation might be
  // possible that accumulates the metric on the fly ...
  typedef MultiImageNCCPostcomputeFilter<
      InputImageType, MetricImageType, GradientImageType, MaskImageType> PostFilterType;

  typename PostFilterType::Pointer postFilter = PostFilterType::New();

  // Configure the post-processing filter
  postFilter->SetComputeGradient(this->m_ComputeGradient);
  postFilter->SetInput(pipeTail->GetOutput());

  // Graft the metric image
  postFilter->GetMetricOutput()->Graft(this->GetMetricOutput());

  // Graft the gradient image if it is needed
  if(this->m_ComputeGradient)
    postFilter->GetGradientOutput()->Graft(this->GetGradientOutput());

  // Set up the weights
  postFilter->SetWeights(this->m_Weights);

  // Set the mask on the post filter
  if(this->GetFixedMaskImage())
    postFilter->SetMaskImage(this->GetFixedMaskImage());

  // Run the post-filter
  postFilter->Update();

#ifdef DUMP_NCC
  // TODO: trash this code!!!!
  // Get and save the metric image
  typename itk::ImageFileWriter<FloatImageType>::Pointer writer = itk::ImageFileWriter<FloatImageType>::New();
  writer->SetInput(postFilter->GetMetricImage());
  writer->SetFileName("nccmap.nii.gz");
  writer->Update();

  typename itk::ImageFileWriter<VectorImageType>::Pointer qwriter = itk::ImageFileWriter<VectorImageType>::New();
  qwriter->SetInput(result);
  qwriter->SetFileName("nccgrad.mha");
  qwriter->Update();
#endif

  // Get the metric
  m_MetricValue = postFilter->GetMetricValue();
}






#endif // __MultiComponentNCCImageMetric_txx

