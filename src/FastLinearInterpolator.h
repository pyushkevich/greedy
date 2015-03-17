#ifndef __FastLinearInterpolator_h_
#define __FastLinearInterpolator_h_

#include "itkVectorImage.h"

/**
 * Base class for the fast linear interpolators
 */
template <class TFloat, unsigned int VDim>
class FastLinearInterpolatorBase
{
public:
  typedef itk::VectorImage<TFloat, VDim>         ImageType;
  typedef TFloat                                 InputComponentType;

  enum InOut { INSIDE, OUTSIDE, BORDER };


  FastLinearInterpolatorBase(ImageType *image)
  {
    buffer = image->GetBufferPointer();
    nComp = image->GetNumberOfComponentsPerPixel();
    def_value_store = vnl_vector<TFloat>(nComp, (TFloat) 0.0);
    def_value = def_value_store.data_block();
  }

protected:


  int nComp;
  const InputComponentType *buffer, *def_value;
  vnl_vector<InputComponentType> def_value_store;
  InOut status;


  inline TFloat lerp(TFloat a, TFloat l, TFloat h)
  {
    return l+((h-l)*a);
  }
};


/**
 * Arbitrary dimension fast linear interpolator - meant to be slow
 */
template <class TFloat, unsigned int VDim>
class FastLinearInterpolator : public FastLinearInterpolatorBase<TFloat, VDim>
{
public:
  typedef FastLinearInterpolatorBase<TFloat, VDim>     Superclass;
  typedef typename Superclass::ImageType               ImageType;
  typedef typename Superclass::InputComponentType      InputComponentType;
  typedef typename Superclass::InOut                   InOut;

  FastLinearInterpolator(ImageType *image) : Superclass(image) {}

  InOut InterpolateWithGradient(float *cix, InputComponentType *out, InputComponentType **grad)
    { return Superclass::INSIDE; }

  InOut Interpolate(float *cix, InputComponentType *out)
    { return Superclass::INSIDE; }

  TFloat GetMask() { return 0.0; }

  TFloat GetMaskAndGradient(TFloat *mask_gradient) { return 0.0; }


protected:
};

/**
 * 3D fast linear interpolator - optimized for speed
 */
template <class TFloat>
class FastLinearInterpolator<TFloat, 3> : public FastLinearInterpolatorBase<TFloat, 3>
{
public:
  typedef FastLinearInterpolatorBase<TFloat, 3>        Superclass;
  typedef typename Superclass::ImageType               ImageType;
  typedef typename Superclass::InputComponentType      InputComponentType;
  typedef typename Superclass::InOut                   InOut;

  FastLinearInterpolator(ImageType *image) : Superclass(image)
  {
    xsize = image->GetLargestPossibleRegion().GetSize()[0];
    ysize = image->GetLargestPossibleRegion().GetSize()[1];
    zsize = image->GetLargestPossibleRegion().GetSize()[2];
  }

  /**
   * Compute the pointers to the eight corners of the interpolating cube
   */
  InOut ComputeCorners(float *cix)
  {
    const InputComponentType *dp;

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
      // The sample point is completely inside
      dp = dens(x0, y0, z0);
      d000 = dp;
      d100 = dp+this->nComp;
      dp += xsize*this->nComp;
      d010 = dp;
      d110 = dp+this->nComp;
      dp += xsize*ysize*this->nComp;
      d011 = dp;
      d111 = dp+this->nComp;
      dp -= xsize*this->nComp;
      d001 = dp;
      d101 = dp+this->nComp;

      // The mask is one
      this->status = Superclass::INSIDE;
      }
    else if (x0 >= -1 && x1 <= xsize &&
             y0 >= -1 && y1 <= ysize &&
             z0 >= -1 && z1 <= zsize)
      {
      // The sample point is on the border region
      d000 = border_check(x0, y0, z0, m000);
      d001 = border_check(x0, y0, z1, m001);
      d010 = border_check(x0, y1, z0, m010);
      d011 = border_check(x0, y1, z1, m011);
      d100 = border_check(x1, y0, z0, m100);
      d101 = border_check(x1, y0, z1, m101);
      d110 = border_check(x1, y1, z0, m110);
      d111 = border_check(x1, y1, z1, m111);

      // The mask is between 0 and 1
      this->status = Superclass::BORDER;
      }
    else
      {
      // The mask is zero
      this->status = Superclass::OUTSIDE;
      }

    return this->status;
  }

  /**
   * Interpolate at position cix, placing the intensity values in out and gradient
   * values in grad (in strides of VDim)
   */
  InOut InterpolateWithGradient(float *cix, InputComponentType *out, InputComponentType **grad)
  {
    double dx00, dx01, dx10, dx11, dxy0, dxy1;
    double dx00_x, dx01_x, dx10_x, dx11_x, dxy0_x, dxy1_x;
    double dxy0_y, dxy1_y;

    // Compute the corners
    this->ComputeCorners(cix);

    if(this->status != Superclass::OUTSIDE)
      {
      // Loop over the components
      for(int iComp = 0; iComp < this->nComp; iComp++, grad++,
          d000++, d001++, d010++, d011++,
          d100++, d101++, d110++, d111++)
        {
        // Interpolate the image intensity
        dx00 = Superclass::lerp(fx, *d000, *d100);
        dx01 = Superclass::lerp(fx, *d001, *d101);
        dx10 = Superclass::lerp(fx, *d010, *d110);
        dx11 = Superclass::lerp(fx, *d011, *d111);
        dxy0 = Superclass::lerp(fy, dx00, dx10);
        dxy1 = Superclass::lerp(fy, dx01, dx11);
        *(out++) = Superclass::lerp(fz, dxy0, dxy1);

        // Interpolate the gradient in x
        dx00_x = *d100 - *d000;
        dx01_x = *d101 - *d001;
        dx10_x = *d110 - *d010;
        dx11_x = *d111 - *d011;
        dxy0_x = this->lerp(fy, dx00_x, dx10_x);
        dxy1_x = this->lerp(fy, dx01_x, dx11_x);
        (*grad)[0] = this->lerp(fz, dxy0_x, dxy1_x);

        // Interpolate the gradient in y
        dxy0_y = dx10 - dx00;
        dxy1_y = dx11 - dx01;
        (*grad)[1] = this->lerp(fz, dxy0_y, dxy1_y);

        // Interpolate the gradient in z
        (*grad)[2] = dxy1 - dxy0;
        }
      }

    return this->status;
  }

  InOut Interpolate(float *cix, InputComponentType *out)
  {
    double dx00, dx01, dx10, dx11, dxy0, dxy1;

    // Compute the corners
    this->ComputeCorners(cix);

    if(this->status != Superclass::OUTSIDE)
      {
      // Loop over the components
      for(int iComp = 0; iComp < this->nComp; iComp++,
          d000++, d001++, d010++, d011++,
          d100++, d101++, d110++, d111++)
        {
        // Interpolate the image intensity
        dx00 = Superclass::lerp(fx, *d000, *d100);
        dx01 = Superclass::lerp(fx, *d001, *d101);
        dx10 = Superclass::lerp(fx, *d010, *d110);
        dx11 = Superclass::lerp(fx, *d011, *d111);
        dxy0 = Superclass::lerp(fy, dx00, dx10);
        dxy1 = Superclass::lerp(fy, dx01, dx11);
        *(out++) = Superclass::lerp(fz, dxy0, dxy1);
        }
      }

    return this->status;
  }

  TFloat GetMask()
  {
    // Interpolate the mask
    double dx00, dx01, dx10, dx11, dxy0, dxy1;
    dx00 = this->lerp(fx, m000, m100);
    dx01 = this->lerp(fx, m001, m101);
    dx10 = this->lerp(fx, m010, m110);
    dx11 = this->lerp(fx, m011, m111);
    dxy0 = this->lerp(fy, dx00, dx10);
    dxy1 = this->lerp(fy, dx01, dx11);
    return this->lerp(fz, dxy0, dxy1);
  }

  TFloat GetMaskAndGradient(TFloat *mask_gradient)
  {
    // Interpolate the mask
    double dx00, dx01, dx10, dx11, dxy0, dxy1;
    dx00 = this->lerp(fx, m000, m100);
    dx01 = this->lerp(fx, m001, m101);
    dx10 = this->lerp(fx, m010, m110);
    dx11 = this->lerp(fx, m011, m111);
    dxy0 = this->lerp(fy, dx00, dx10);
    dxy1 = this->lerp(fy, dx01, dx11);
    double mask = this->lerp(fz, dxy0, dxy1);

    // Compute the gradient of the mask
    double dx00_x, dx01_x, dx10_x, dx11_x, dxy0_x, dxy1_x;
    dx00_x = m100 - m000;
    dx01_x = m101 - m001;
    dx10_x = m110 - m010;
    dx11_x = m111 - m011;
    dxy0_x = this->lerp(fy, dx00_x, dx10_x);
    dxy1_x = this->lerp(fy, dx01_x, dx11_x);
    mask_gradient[0] = this->lerp(fz, dxy0_x, dxy1_x);

    double dxy0_y, dxy1_y;
    dxy0_y = dx10 - dx00;
    dxy1_y = dx11 - dx01;
    mask_gradient[1] = this->lerp(fz, dxy0_y, dxy1_y);

    mask_gradient[2] = dxy1 - dxy0;

    return mask;
  }

protected:

  inline const InputComponentType *border_check(int X, int Y, int Z, InputComponentType &mask)
  {
    if(X >= 0 && X < xsize && Y >= 0 && Y < ysize && Z >= 0 && Z < zsize)
      {
      mask = 1.0;
      return dens(X,Y,Z);
      }
    else
      {
      mask = 0.0;
      return this->def_value;
      }
   }

  inline const InputComponentType *dens(int X, int Y, int Z)
  {
    return this->buffer + this->nComp * (X+xsize*(Y+ysize*Z));
  }

  // Image size
  int xsize, ysize, zsize;

  // State of current interpolation
  const InputComponentType *d000, *d001, *d010, *d011, *d100, *d101, *d110, *d111;
  InputComponentType m000, m001, m010, m011, m100, m101, m110, m111;

  double fx, fy, fz;
  int	 x0, y0, z0, x1, y1, z1;

};

#endif
