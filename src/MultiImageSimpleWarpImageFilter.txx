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
namespace itk
{

/**
 * Default constructor.
 */

template <class TImage, class TVectorImage, class TFloat>
MultiImageOpticalFlowImageFilter<TImage,TVectorImage,TFloat>
::MultiImageOpticalFlowImageFilter()
{
  // Setup default values
  m_Deformation = NULL;
  m_EdgePaddingValue = NumericTraits<PixelType>::Zero;
}

/**
 * Standard PrintSelf method.
 */
template <class TImage, class TVectorImage, class TFloat>
void
MultiImageOpticalFlowImageFilter<TImage,TVectorImage,TFloat>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

/**
 * Set deformation field as Inputs[1] for this ProcessObject.
 *
 */
template <class TImage, class TVectorImage, class TFloat>
void
MultiImageOpticalFlowImageFilter<TImage,TVectorImage,TFloat>
::SetDeformationField(
  const DeformationFieldType * field )
{
  // const cast is needed because the pipeline is not const-correct.
  DeformationFieldType * input =  
       const_cast< DeformationFieldType * >( field );
  this->ProcessObject::SetInput("deformation", input);
  m_Deformation = field;
}

template <class TImage, class TVectorImage, class TFloat>
void
MultiImageOpticalFlowImageFilter<TImage,TVectorImage,TFloat>
::AddImagePair(InputImageType *fixed,
               InputImageType *moving,
               InputVectorImageType *gradMoving,
               double weight)
{
  // Store the image ourselves
  ImageSet is;
  is.fixed = fixed;
  is.moving = moving;
  is.grad_moving = gradMoving;
  is.weight = weight;
  m_ImageSet.push_back(is);

  // Set as inputs
  this->ProcessObject::SetInput(std::string("moving_") + m_ImageSet.size(), moving);
  this->ProcessObject::SetInput(std::string("fixed_") + m_ImageSet.size(), fixed);
  this->ProcessObject::SetInput(std::string("gradMoving_") + m_ImageSet.size(), gradMoving);
}

/**
 * Return a pointer to the deformation field.
 */
template <class TImage, class TVectorImage, class TFloat>
typename MultiImageOpticalFlowImageFilter<TImage,TVectorImage,TFloat>::DeformationFieldType *
MultiImageOpticalFlowImageFilter<TImage,TVectorImage,TFloat>
::GetDeformationField(void)
{
  return m_Deformation;
}


/**
 * Setup state of filter before multi-threading.
 * InterpolatorType::SetInputImage is not thread-safe and hence
 * has to be setup before ThreadedGenerateData
 */
template <class TImage, class TVectorImage, class TFloat>
void
MultiImageOpticalFlowImageFilter<TImage,TVectorImage,TFloat>
::BeforeThreadedGenerateData()
{
  // What to do here?
}

/**
 * Setup state of filter after multi-threading.
 */
template <class TImage, class TVectorImage, class TFloat>
void
MultiImageOpticalFlowImageFilter<TImage,TVectorImage,TFloat>
::AfterThreadedGenerateData()
{

}

/**
  Trilinear interpolation
 */
void trilerp()
{
#define DENS(X, Y, Z) ((X)+xsize*((Y)+ysize*(Z)))
#define INRANGE(X, Y, Z) ((X) >= 0 && (X) < xsize && (Y) >= 0 && (Y) < ysize && (Z) >= 0 && (Z) < zsize)


    int	       x0, y0, z0, x1, y1, z1, dp;
    int        d000, d001, d010, d011,
               d100, d101, d110, d111;

    double     fx, fy, fz,
            d000, d001, d010, d011,
            d100, d101, d110, d111,
            dx00, dx01, dx10, dx11,
            dxy0, dxy1, dxyz;

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
        dp = DENS(x0, y0, z0);
        d000 = dp;
        d100 = dp+1;
        dp += xsize;
        d010 = dp;
        d110 = dp+1;
        dp += xsize*ysize;
        d011 = dp;
        d111 = dp+1;
        dp -= xsize;
        d001 = dp;
        d101 = dp+1;
        }
    else
        {
        d000 = INRANGE(x0, y0, z0) ? DENS(x0, y0, z0) : def;
        d001 = INRANGE(x0, y0, z1) ? DENS(x0, y0, z1) : def;
        d010 = INRANGE(x0, y1, z0) ? DENS(x0, y1, z0) : def;
        d011 = INRANGE(x0, y1, z1) ? DENS(x0, y1, z1) : def;
        d100 = INRANGE(x1, y0, z0) ? DENS(x1, y0, z0) : def;
        d101 = INRANGE(x1, y0, z1) ? DENS(x1, y0, z1) : def;
        d110 = INRANGE(x1, y1, z0) ? DENS(x1, y1, z0) : def;
        d111 = INRANGE(x1, y1, z1) ? DENS(x1, y1, z1) : def;
        }

    // Loop over the moving images
    for(int i = 0; i < nPair; i++)
        {

        }
    dx00 = LERP(fx, d000, d100);
    dx01 = LERP(fx, d001, d101);
    dx10 = LERP(fx, d010, d110);
    dx11 = LERP(fx, d011, d111);

    dxy0 = LERP(fy, dx00, dx10);
    dxy1 = LERP(fy, dx01, dx11);

    dxyz = LERP(fz, dxy0, dxy1);

    return dxyz;
}



/**
 * Compute the output for the region specified by outputRegionForThread.
 */
template <class TImage, class TVectorImage, class TFloat>
void
MultiImageOpticalFlowImageFilter<TImage,TVectorImage,TFloat>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  ThreadIdType threadId )
{
  // Loop over the deformation field
  typedef ImageRegionIteratorWithIndex<DeformationFieldType> DefIterType;
  DefIterType it_def(m_Deformation, outputRegionForThread);

  while (!it_def.IsAtEnd())
    {
    // Compute the position where the moving images are going to be interpolated
    IndexType &index = it_def.GetIndex();
    DisplacementType &disp = it_def.Get();
    itk::ContinuousIndex<TFloat,ImageDimension> cix;
    for(unsigned int j = 0; j < ImageDimension; j++ )
      cix[j] = index[j] + m_DeformationScaling * disp[j];

    // Compute the indices of the sampling points
    ComputeIndex<3>(mov_size, index);





    }


  InputImageConstPointer inputPtr = this->GetInput();
  OutputImagePointer outputPtr = this->GetOutput();
  DeformationFieldPointer fieldPtr = this->GetDeformationField();

  // iterator for the output image
  ImageRegionIteratorWithIndex<OutputImageType> outputIt(
    outputPtr, outputRegionForThread );
  IndexType index;
  itk::ContinuousIndex<TFloat,ImageDimension> cix;
  DisplacementType displacement;
  
  // iterator for the deformation field
  ImageRegionIterator<DeformationFieldType> 
    fieldIt(fieldPtr, outputRegionForThread );

  while( !outputIt.IsAtEnd() )
    {
    // get the output image index
    index = outputIt.GetIndex();

    // get the required displacement
    displacement = fieldIt.Get();

    // compute the required input image point
    for(unsigned int j = 0; j < ImageDimension; j++ )
      {
      cix[j] = index[j] + m_DeformationScaling * displacement[j];
      }

    // get the interpolated value
    if( m_Interpolator->IsInsideBuffer( cix ) )
      {
      PixelType value = 
        static_cast<PixelType>(m_Interpolator->EvaluateAtContinuousIndex( cix ) );
      outputIt.Set( value );
      }
    else
      {
      outputIt.Set( m_EdgePaddingValue );
      }   
    ++outputIt;
    ++fieldIt; 
    }
}


template <class TInputImage,class TOutputImage,class TDeformationField, class TFloat>
void
SimpleWarpImageFilter<TInputImage,TOutputImage,TDeformationField,TFloat>
::GenerateInputRequestedRegion()
{

  // call the superclass's implementation
  Superclass::GenerateInputRequestedRegion();

  // request the largest possible region for the input image
  InputImagePointer inputPtr = 
    const_cast< InputImageType * >( this->GetInput() );

  if( inputPtr )
    {
    inputPtr->SetRequestedRegionToLargestPossibleRegion();
    }

  // just propagate up the output requested region for the 
  // deformation field.
  DeformationFieldPointer fieldPtr = this->GetDeformationField();
  OutputImagePointer outputPtr = this->GetOutput();
  if(fieldPtr.IsNotNull() )
    {
    fieldPtr->SetRequestedRegion( outputPtr->GetRequestedRegion() );
    if(!fieldPtr->VerifyRequestedRegion())
      {
      fieldPtr->SetRequestedRegion(fieldPtr->GetLargestPossibleRegion());
      }
    }
}


template <class TInputImage,class TOutputImage,class TDeformationField, class TFloat>
void
SimpleWarpImageFilter<TInputImage,TOutputImage,TDeformationField,TFloat>
::GenerateOutputInformation()
{
  // call the superclass's implementation of this method
  Superclass::GenerateOutputInformation();

  OutputImagePointer outputPtr = this->GetOutput();
  DeformationFieldPointer fieldPtr = this->GetDeformationField();
  outputPtr->SetSpacing( fieldPtr->GetSpacing() );
  outputPtr->SetOrigin( fieldPtr->GetOrigin() );
  outputPtr->SetDirection( fieldPtr->GetDirection() );
  outputPtr->SetLargestPossibleRegion( fieldPtr->
                                       GetLargestPossibleRegion() );
}


} // end namespace itk

#endif
