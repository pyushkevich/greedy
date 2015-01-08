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

  InOut Interpolate(float *cix, int stride, InputComponentType *out)
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

  InOut Interpolate(float *cix, int stride, InputComponentType *out)
  {

    const InputComponentType *dp;

    double fx, fy, fz;
    double dx00, dx01, dx10, dx11, dxy0, dxy1;

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
      d000 = inrange(x0, y0, z0) ? dens(x0, y0, z0) : this->def_value;
      d001 = inrange(x0, y0, z1) ? dens(x0, y0, z1) : this->def_value;
      d010 = inrange(x0, y1, z0) ? dens(x0, y1, z0) : this->def_value;
      d011 = inrange(x0, y1, z1) ? dens(x0, y1, z1) : this->def_value;
      d100 = inrange(x1, y0, z0) ? dens(x1, y0, z0) : this->def_value;
      d101 = inrange(x1, y0, z1) ? dens(x1, y0, z1) : this->def_value;
      d110 = inrange(x1, y1, z0) ? dens(x1, y1, z0) : this->def_value;
      d111 = inrange(x1, y1, z1) ? dens(x1, y1, z1) : this->def_value;

      // The mask is between 0 and 1
      this->status = Superclass::BORDER;
      }
    else
      {
      // The sample point is outside
      for(int iComp = 0; iComp < this->nComp; iComp+=stride)
        *(out++) = this->def_value[iComp];

      // The mask is zero
      return Superclass::OUTSIDE;
      }

    // Interpolate each component
    for(int iComp = 0; iComp < this->nComp; iComp+=stride,
        d000+=stride, d001+=stride, d010+=stride, d011+=stride,
        d100+=stride, d101+=stride, d110+=stride, d111+=stride)
      {
      // Interpolate first component
      dx00 = Superclass::lerp(fx, *d000, *d100);
      dx01 = Superclass::lerp(fx, *d001, *d101);
      dx10 = Superclass::lerp(fx, *d010, *d110);
      dx11 = Superclass::lerp(fx, *d011, *d111);
      dxy0 = Superclass::lerp(fy, dx00, dx10);
      dxy1 = Superclass::lerp(fy, dx01, dx11);
      *(out++) = Superclass::lerp(fz, dxy0, dxy1);
      }

    return this->status;
  }

  TFloat GetMask()
  {
    // Interpolate the mask
    double dx00, dx01, dx10, dx11, dxy0, dxy1;
    dx00 = this->lerp(fx, d000 == this->def_value ? 0.0 : 1.0, d100 == this->def_value ? 0.0 : 1.0);
    dx01 = this->lerp(fx, d001 == this->def_value ? 0.0 : 1.0, d101 == this->def_value ? 0.0 : 1.0);
    dx10 = this->lerp(fx, d010 == this->def_value ? 0.0 : 1.0, d110 == this->def_value ? 0.0 : 1.0);
    dx11 = this->lerp(fx, d011 == this->def_value ? 0.0 : 1.0, d111 == this->def_value ? 0.0 : 1.0);
    dxy0 = this->lerp(fy, dx00, dx10);
    dxy1 = this->lerp(fy, dx01, dx11);
    return this->lerp(fz, dxy0, dxy1);
  }

  TFloat GetMaskAndGradient(TFloat *mask_gradient)
  {
    // Compute the gradient of the mask
    mask_gradient[0] = (x0 == 0) ? 1.0 : ((x1 == xsize) ? -1.0 : 0.0);
    mask_gradient[1] = (y0 == 0) ? 1.0 : ((y1 == ysize) ? -1.0 : 0.0);
    mask_gradient[2] = (z0 == 0) ? 1.0 : ((z1 == zsize) ? -1.0 : 0.0);

    // Interpolate the mask
    double dx00, dx01, dx10, dx11, dxy0, dxy1;
    dx00 = this->lerp(fx, d000 == this->def_value ? 0.0 : 1.0, d100 == this->def_value ? 0.0 : 1.0);
    dx01 = this->lerp(fx, d001 == this->def_value ? 0.0 : 1.0, d101 == this->def_value ? 0.0 : 1.0);
    dx10 = this->lerp(fx, d010 == this->def_value ? 0.0 : 1.0, d110 == this->def_value ? 0.0 : 1.0);
    dx11 = this->lerp(fx, d011 == this->def_value ? 0.0 : 1.0, d111 == this->def_value ? 0.0 : 1.0);
    dxy0 = this->lerp(fy, dx00, dx10);
    dxy1 = this->lerp(fy, dx01, dx11);
    return this->lerp(fz, dxy0, dxy1);
  }

protected:

  inline bool inrange(int X, int Y, int Z)
  {
    return
        X >= 0 && X < xsize &&
        Y >= 0 && Y < ysize &&
        Z >= 0 && Z < zsize;
  }

  inline const InputComponentType *dens(int X, int Y, int Z)
  {
    return this->buffer + this->nComp * (X+xsize*(Y+ysize*Z));
  }

  // Image size
  int xsize, ysize, zsize;

  // State of current interpolation
  const InputComponentType *d000, *d001, *d010, *d011, *d100, *d101, *d110, *d111;
  double fx, fy, fz;
  int	 x0, y0, z0, x1, y1, z1;

};

#endif
