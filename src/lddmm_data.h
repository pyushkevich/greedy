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
#ifndef _LDDMM_DATA_H_
#define _LDDMM_DATA_H_

#include <lddmm_common.h>
#include <itkNumericTraits.h>
#include <itkImage.h>
#include <itkVectorImage.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkCovariantVector.h>
#include <itkMatrix.h>
#include <vnl/vnl_math.h>
#include <vector>

#include <itkImageIOBase.h>

#ifdef _LDDMM_FFT_
#include <fftw3.h>
#endif

template<class TFloat, uint VDim>
class LDDMMData
{
public:
  // Image data
  typedef itk::ImageBase<VDim> ImageBaseType;
  typedef typename ImageBaseType::Pointer ImageBasePointer;
  typedef itk::Image<TFloat, VDim> ImageType;
  typedef typename ImageType::Pointer ImagePointer;
  typedef itk::ImageRegionIteratorWithIndex<ImageType> ImageIterator;

  // Vector fields, etc
  typedef itk::CovariantVector<TFloat, VDim> Vec;
  typedef itk::Image<Vec, VDim> VectorImageType;
  typedef typename VectorImageType::Pointer VectorImagePointer;
  typedef std::vector<VectorImagePointer> VelocityField;

  // Jacobians
  typedef itk::Matrix<TFloat, VDim, VDim> Mat;
  typedef itk::Image<Mat, VDim> MatrixImageType;
  typedef typename MatrixImageType::Pointer MatrixImagePointer;

  // Composite images (variable number of components)
  typedef itk::VectorImage<TFloat, VDim> CompositeImageType;
  typedef typename CompositeImageType::Pointer CompositeImagePointer;

  // Regions, etc
  typedef itk::ImageRegion<VDim> RegionType;

  // Pointers to the fixed and moving images
  ImagePointer fix, mov;

  // Fourier space kernels
  ImagePointer f_kernel, f_kernel_sq;

  // Velocity field pointers (v, phi, a used for semi-lagrange scheme)
  VelocityField v, f, a;

  // Region for the velocity fields
  RegionType r;

  // Parameters
  double alpha, sigma, gamma, dt, sigma_sq;
  
  // Dimensions
  uint n[VDim];

  // Number of timesteps, number of voxels
  uint nt, nv;

  // Allocate image using an existing pointer
  static void alloc_img(ImageType *img, ImageBaseType *ref);
  static void alloc_vimg(VectorImageType *vimg, ImageBaseType *ref);
  static void alloc_cimg(CompositeImageType *img, ImageBaseType *ref, int n_comp);
  static void alloc_mimg(MatrixImageType *img, ImageBaseType *ref);

  // These methods are shorthands for creating an image and allocating it
  static ImagePointer new_img(ImageBaseType *ref);
  static VectorImagePointer new_vimg(ImageBaseType *ref);
  static MatrixImagePointer new_mimg(ImageBaseType *ref);
  static CompositeImagePointer new_cimg(ImageBaseType *ref, int n_comp);

  // Create and allocate velocity field
  static void new_vf(VelocityField &vf, uint nt, ImageBaseType *ref);

  // Initialize LDDMM data 
  static void init(LDDMMData<TFloat, VDim> &, 
    ImageType *fix, ImageType *mov, 
    uint nt, double alpha, double gamma, double sigma);

  // Apply deformation to data
  static void interp_vimg(
    VectorImageType *data, VectorImageType *field, 
    TFloat def_scale, VectorImageType *out,
    bool use_nn = false, bool phys_space = false);

  // Apply deformation to data
  static void interp_img(ImageType *data, VectorImageType *field, ImageType *out,
                         bool use_nn = false, bool phys_space = false, TFloat outside_value = 0.0);

  // Apply deformation to data
  static void interp_cimg(CompositeImageType *data, VectorImageType *field, CompositeImageType *out,
                          bool use_nn = false, bool phys_space = false, TFloat outside_value = 0.0);

  // Apply deformation to matrix data
  static void interp_mimg(MatrixImageType *data, VectorImageType *field, MatrixImageType *out,
                          bool use_nn = false, bool phys_space = false);

  // Compute the Jacobian of the field, to be placed into an array of vector images
  // The output Jacobians are stored 
  static void field_jacobian(VectorImageType *vec, MatrixImageType *out);

  // Divergence of a vector field
  static void field_divergence(VectorImageType *v, ImageType *div_v, bool use_spacing);

  // Jacobian of warp composition. Let f(x) = x + u(x) and g(x) = x + v(x) and let h = x + w(x)
  // Suppose h(x) = f(g(x)), then w(x) = v(x) + u(x + v(x))
  // This code computes the Jacobian of w, Dw, given the Jacobians of u and v, as well as v itself.
  // Here jac_phi and jac_psi may be the same image, but they should both be different from
  // the output image. 
  //
  // Keep in mind that Dw is not the same as Dh (there is a difference of identity matrix). However
  // when composing warp jacobians it is better to work with Du and Dv than with Df and Dg because 
  // interpolation at the boundary fills unknown values with zeros
  static void jacobian_of_composition(
    MatrixImageType *Du, MatrixImageType *Dv, VectorImageType *v, MatrixImageType *out_Dw);

  // Compute a determinant of M + \lambda I
  static void mimg_det(MatrixImageType *M, double lambda, ImageType *out_det);

  // Compute (lambda) Ax + (mu) b and store in (out). Out may point to x or b for an in-place operation
  static void mimg_vimg_product_plus_vimg(
    MatrixImageType *A, VectorImageType *x, VectorImageType *b, 
    TFloat lambda, TFloat mu, VectorImageType *out);

  // Take Jacobian of deformation field
  static void field_jacobian_det(VectorImageType *vec, ImageType *out);

  // Compute the Lie bracket by computing two Jacobians and multiplying. This should be made faster
  // by making a custom ITK filter
  static void lie_bracket(VectorImageType *v, VectorImageType *u, MatrixImageType *work, VectorImageType *out);

  // Smooth an image in-place
  static void img_smooth(ImageType *src, ImageType *out, Vec sigma);
  static void vimg_smooth(VectorImageType *src, VectorImageType *out, double sigma);
  static void vimg_smooth(VectorImageType *src, VectorImageType *out, Vec sigmas);
  static void cimg_smooth(CompositeImageType *src, CompositeImageType *out, Vec sigma);

  // Smooth a displacement field with a border of zeros around it
  static void vimg_smooth_withborder(VectorImageType *src, VectorImageType *trg, Vec sigma, int border_size);

  // Take gradient of an image
  static void image_gradient(ImageType *src, VectorImageType *grad, bool use_spacing);

  // Basic math
  static void vimg_add_in_place(VectorImageType *trg, VectorImageType *a);
  static void vimg_subtract_in_place(VectorImageType *trg, VectorImageType *a);
  static void vimg_scale_in_place(VectorImageType *trg, TFloat s);

  // compute trg = trg + s * a
  static void vimg_add_scaled_in_place(VectorImageType *trg, VectorImageType *a, TFloat s);

  static void vimg_scale(const VectorImageType *src, TFloat s, VectorImageType *trg);
  static void vimg_multiply_in_place(VectorImageType *trg, ImageType *s);
  static void vimg_euclidean_inner_product(ImagePointer &trg, VectorImageType *a, VectorImageType *b);
  static TFloat vimg_euclidean_norm_sq(VectorImageType *trg);

  // Matrix - matrix multiplication
  static void mimg_multiply_in_place(MatrixImageType *trg, MatrixImageType *s);

  // Compute the range of the norm of a vector field
  static void vimg_norm_min_max(VectorImageType *image, ImageType *normsqr,
    TFloat &min_norm, TFloat &max_norm);

  // Update a vector image to make its maxumum length equal to given value. The
  // second parameter is a working image that will return unnormalized lengths squared
  static void vimg_normalize_to_fixed_max_length(VectorImageType *trg,
                                                 ImageType *normsqr,
                                                 double max_displacement,
                                                 bool scale_down_only);

  // Scalar math
  static void img_add_in_place(ImageType *trg, ImageType *a);
  static void img_subtract_in_place(ImageType *trg, ImageType *a);
  static void img_multiply_in_place(ImageType *trg, ImageType *a);
  static void img_scale_in_place(ImageType *trg, TFloat scale);
  static TFloat img_euclidean_norm_sq(ImageType *trg);
  static TFloat img_voxel_sum(ImageType *trg);
  static void img_min_max(ImageType *src, TFloat &out_min, TFloat &out_max);

  // Generate a kernel image for navier-stokes operator
  static void compute_navier_stokes_kernel(ImageType *kernel, double alpha, double gamma);

  // Create a reference space (unallocated) for downsampling. Mostly used internally
  static ImageBasePointer create_reference_space_for_downsample(ImageBaseType *src, double factor);

  // New, hopefully faster, downsample code
  static ImagePointer img_downsample(ImageType *src, double factor);
  static CompositeImagePointer cimg_downsample(CompositeImageType *img, double factor);

  // Downsample and upsample images (includes smoothing, use sparingly)
  static void img_shrink(ImageType *src, ImageType *trg, int factor);
  static void img_resample_identity(ImageType *src, ImageBaseType *ref, ImageType *trg);
  static void vimg_resample_identity(VectorImageType *src, ImageBaseType *ref, VectorImageType *trg);

  // Versions of these methods that return pointers
  static VectorImagePointer vimg_resample_identity(VectorImageType *src, ImageBaseType *ref);

  // Threshold image
  static void img_threshold_in_place(ImageType *src, double lt, double up, double fore, double back);

  // Filter NaNs from an image
  static void img_filter_nans_in_place(ImageType *src, ImageType *nan_mask);

  // Replace NaNs in an image using a mask
  static void img_reconstitute_nans_in_place(ImageType *src, ImageType *nan_mask);

  // Composite image matj
  static void cimg_add_in_place(CompositeImageType *trg, CompositeImageType *a);
  static void cimg_scale_in_place(CompositeImageType *trg, TFloat scale);
  static void cimg_threshold_in_place(CompositeImageType *trg,
                                      double lt, double up, double fore, double back);

  // Extract single component in cimg
  static void cimg_extract_component(CompositeImageType *src, ImageType *trg, unsigned int c);
  static void cimg_update_component(CompositeImageType *src, ImageType *trg, unsigned int c);

  // Apply noise to image components
  static void cimg_add_gaussian_noise_in_place(CompositeImageType *img,
                                               const std::vector<double> &sigma,
                                               unsigned long stride = 0);

  static void vimg_add_gaussian_noise_in_place(VectorImageType *img,
                                               double sigma,
                                               unsigned long stride = 0);

  // Convert voxel-space warp to a physical space warp
  static void warp_voxel_to_physical(VectorImageType *src, ImageBaseType *ref_space, VectorImageType *trg);
  
  // Some IO methods
  typedef itk::ImageIOBase::IOComponentEnum IOComponentType;

  static IOComponentType img_read(const char *fn, ImagePointer &trg);
  static IOComponentType vimg_read(const char *fn, VectorImagePointer &trg);
  static IOComponentType cimg_read(const char *fn, CompositeImagePointer &trg);

  static ImagePointer img_read(const char *fn);
  static VectorImagePointer vimg_read(const char *fn);
  static CompositeImagePointer cimg_read(const char *fn);

  // Write scalar image, with optional output format specification
  static void img_write(ImageType *src, const char *fn,
                        IOComponentType comp = itk::IOComponentEnum::UNKNOWNCOMPONENTTYPE);

  // Write vector image, with optional output format specification
  static void vimg_write(VectorImageType *src, const char *fn,
                         IOComponentType comp = itk::IOComponentEnum::UNKNOWNCOMPONENTTYPE);

  // Write composite image, with optional output format specification
  static void cimg_write(CompositeImageType *src, const char *fn,
                         IOComponentType comp = itk::IOComponentEnum::UNKNOWNCOMPONENTTYPE);

  static void vfield_read(uint nt, const char *fnpat, VelocityField &v);

  static void vimg_copy(const VectorImageType *src, VectorImageType *trg);
  static void img_copy(const ImageType *src, ImageType *trg);
  static void mimg_copy(const MatrixImageType *src, MatrixImageType *trg);
  static void cimg_copy(const CompositeImageType *src, CompositeImageType *trg);

  /** Cast to image of different datatype automatically */
  static bool vimg_auto_cast(const VectorImageType *src, ImageBaseType *trg);
  static bool img_auto_cast(const ImageType *src, ImageBaseType *trg);
  static bool cimg_auto_cast(const CompositeImageType *src, ImageBaseType *trg);

  // Check if two images occupy same space
  static bool img_same_space(const ImageBaseType *i1, const ImageBaseType *i2, double tol = 1e-6);

  // Merge a bunch of images into one. If the input array only has one image, it will be returned
  static CompositeImagePointer cimg_concat(const std::vector<CompositeImagePointer> &img);

  // Count the number of NaNs in an image
  static unsigned int cimg_nancount(const CompositeImageType *img);

  // Compute (x / m if m >= t, 0 o/w)
  static void cimg_mask_smooth_adjust_in_place(CompositeImageType *img, ImageType *mask, TFloat thresh);

  // Compute a array from v
  void compute_semi_lagrangean_a();

  // Exponentiate a deformation field using scaling and recursive squaring. 
  // The source image may not be the same as the target image!
  static void vimg_exp(
    const VectorImageType *src, VectorImageType *trg, VectorImageType *work,
    int exponent, TFloat scale = 1.0);

  // Exponentiate a deformation field and its Jacobian
  static void vimg_exp_with_jacobian(
    const VectorImageType *src, VectorImageType *trg, VectorImageType *work,
    MatrixImageType *trg_jac, MatrixImageType *work_mat,
    int exponent, TFloat scale);

  // Integrate forward tranform (phi_0_t)
  void integrate_phi_t0();
  void integrate_phi_t1();

  // Rectifier functions that begin linear-like and switch to constant-like at the 
  // specified threshold. These should be applied to non-negative quantities
  static void img_linear_to_const_rectifier_fn(ImageType *src, ImageType *trg, TFloat thresh);
  static void img_linear_to_const_rectifier_deriv(ImageType *src, ImageType *trg, TFloat thresh);

  // PDE support (for incompressibility)
  static void *poisson_pde_zero_boundary_initialize(ImageBaseType *ref, ImageType *mask = NULL);
  static void poisson_pde_zero_boundary_solve(void *solver_data, ImageType *rhs, ImageType *solution);
  static void poisson_pde_zero_boundary_laplacian(void *solver_data, ImageType *u, ImageType *result);
  static void poisson_pde_zero_boundary_dealloc(void *solver_data);

protected:

  // Present a vector field as a cimg (for algorithms that only run on cimg). The two images will
  // share memory, and this is meant to be used temporarily
  static CompositeImagePointer vimg_as_cimg(VectorImageType *src);
  static ImagePointer cimg_as_img(CompositeImageType *src);
  static CompositeImagePointer img_as_cimg(ImageType *src);

  // Functor ops for composite images
  template <class TFunctor>
  static void cimg_apply_binary_functor_in_place(CompositeImageType *trg, CompositeImageType *a);

  // A vector image for in-place interpolation operations
  VectorImagePointer vtmp;

};

#ifdef _LDDMM_FFT_

template <class TFloat, uint VDim>
class LDDMMFFTInterface
{
public:
  typedef typename LDDMMData<TFloat, VDim>::ImageType ImageType;
  typedef typename LDDMMData<TFloat, VDim>::VectorImageType VectorImageType;
  typedef typename LDDMMData<TFloat, VDim>::Vec Vec;

  LDDMMFFTInterface(ImageType *ref);
  ~LDDMMFFTInterface();

  void convolution_fft(
    VectorImageType *img, ImageType *kernel_ft, bool inv_kernel, 
    VectorImageType *out);

private:

  // Size of the input array and allocated array (bigger, for in-place math)
  itk::Size<VDim> m_Size, m_Alloc;
  uint m_AllocSize, m_DataSize;

  // In-place data array
  double *m_Data;

  // FFT plan
  fftw_plan m_Plan, m_InvPlan;

};

#else

template <class TFloat, uint VDim>
class LDDMMFFTInterface
{
public:
  typedef typename LDDMMData<TFloat, VDim>::ImageType ImageType;
  typedef typename LDDMMData<TFloat, VDim>::VectorImageType VectorImageType;
  typedef typename LDDMMData<TFloat, VDim>::Vec Vec;

  LDDMMFFTInterface(ImageType *ref) {}
  ~LDDMMFFTInterface() {}

   void convolution_fft(
       VectorImageType *img, ImageType *kernel_ft, bool inv_kernel,
       VectorImageType *out) { throw std::string("Code was not compiled with _LDDMM_FFT_"); }

};

#endif // _LDDMM_FFT_


// Class for iteratively computing the objective function
template<class TFloat, uint VDim>
class LDDMMImageMatchingObjective 
{
public:
  typedef LDDMMData<TFloat, VDim> LDDMM;
  typedef LDDMMFFTInterface<TFloat, VDim> FFT;

  LDDMMImageMatchingObjective(LDDMM &p);
  TFloat compute_objective_and_gradient(LDDMM &p);

  typename LDDMM::ImagePointer Jt0, Jt1, DetPhit1;
  typename LDDMM::VectorImagePointer GradJt0;
  FFT fft;
};




#endif
