#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

#include "lddmm_common.h"
#include "lddmm_data.h"

#include <itkImageFileReader.h>
#include <itkGaussianInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkIdentityTransform.h>
#include <itkShrinkImageFilter.h>

#include "MultiImageSimpleWarpImageFilter.h"
#include <vnl/vnl_cost_function.h>


int usage()
{
  printf("greedy: Paul's greedy diffeomorphic registration implementation\n");
  printf("Usage: \n");
  printf("  greedy [options]\n");
  printf("Required options: \n");
  printf("  -d DIM                      : Number of image dimensions\n");
  printf("  -i fixed.nii moving.nii     : Image pair (may be repeated)\n");
  printf("  -o output.nii               : Output file\n");
  printf("Optional: \n");
  printf("  -a                          : Perform affine registration and save to output (-o)\n");
  printf("  -w weight                   : weight of the next -i pair\n");
  printf("  -e epsilon                  : step size (default = 1.0)\n");
  printf("  -s sigma1 sigma2            : smoothing for the greedy update step (3.0, 1.0)\n");
  printf("  -n NxNxN                    : number of iterations per level of multi-res (100x100) \n");
  printf("  -dump-moving                : dump moving image at each iter\n");
  printf("  -dump-freq N                : dump frequency\n");
  return -1;
}

struct ImagePairSpec
{
  std::string fixed;
  std::string moving;
  double weight;
};

struct GreedyParameters
{
  std::vector<ImagePairSpec> inputs;
  std::string output;
  unsigned int dim; 

  bool flag_optimize_affine;
  bool flag_dump_moving;
  int dump_frequency;
  double epsilon, sigma_pre, sigma_post;

  // Iterations per level (i.e., 40x40x100)
  std::vector<int> iter_per_level;
};


template <unsigned int VDim, typename TReal = double>
class GreedyApproach
{
public:

  typedef LDDMMData<TReal, VDim> LDDMMType;
  typedef typename LDDMMType::ImageBaseType ImageBaseType;
  typedef typename LDDMMType::ImageType ImageType;
  typedef typename LDDMMType::ImagePointer ImagePointer;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::VectorImagePointer VectorImagePointer;
  typedef typename LDDMMType::CompositeImageType CompositeImageType;
  typedef typename LDDMMType::CompositeImagePointer CompositeImagePointer;

  typedef MultiImageOpticalFlowHelper<TReal, VDim> OFHelperType;

  struct ImagePair {
    ImagePointer fixed, moving;
    VectorImagePointer grad_moving;
    double weight;
  };


  static int Run(GreedyParameters &param);

  static int RunAffine(GreedyParameters &param);

protected:

  static void ReadImages(GreedyParameters &param, OFHelperType &ofhelper);
  static void ReadImages(GreedyParameters &param, std::vector<ImagePair> &imgRaw);
  static void ResampleImages(GreedyParameters &param,
                             const std::vector<ImagePair> &imgRaw,
                             std::vector<ImagePair> &img,
                             int level);


  /** Cost function used for conjugate gradient descent */
  class AffineCostFunction : public vnl_cost_function
  {
  public:
    virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g);
  protected:
    std::vector<ImagePair> &img;
  };
};

template <unsigned int VDim, typename TReal>
void
GreedyApproach<VDim, TReal>::AffineCostFunction
::compute(const vnl_vector<double> &x, double *f, vnl_vector<double> *g)
{
  // Form a matrix/vector from x

  // Interpolate the images in img at positions specified by x

}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::ReadImages(GreedyParameters &param, OFHelperType &ofhelper)
{
  // Read the input images and stick them into an image array
  for(int i = 0; i < param.inputs.size(); i++)
    {
    // Read fixed
    typedef itk::ImageFileReader<CompositeImageType> ReaderType;
    typename ReaderType::Pointer readfix = ReaderType::New();
    readfix->SetFileName(param.inputs[i].fixed);
    readfix->Update();

    // Read moving
    typedef itk::ImageFileReader<CompositeImageType> ReaderType;
    typename ReaderType::Pointer readmov = ReaderType::New();
    readmov->SetFileName(param.inputs[i].moving);
    readmov->Update();

    // Add to the helper object
    ofhelper.AddImagePair(readfix->GetOutput(), readmov->GetOutput(), param.inputs[i].weight);
    }
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::ReadImages(GreedyParameters &param, std::vector<ImagePair> &imgRaw)
{
  // Read the input images and stick them into an image array
  for(int i = 0; i < param.inputs.size(); i++)
    {
    ImagePair ip;

    // Read fixed
    typedef itk::ImageFileReader<ImageType> ReaderType;
    typename ReaderType::Pointer readfix = ReaderType::New();
    readfix->SetFileName(param.inputs[i].fixed);
    readfix->Update();
    ip.fixed = readfix->GetOutput();

    // Read moving
    typedef itk::ImageFileReader<ImageType> ReaderType;
    typename ReaderType::Pointer readmov = ReaderType::New();
    readmov->SetFileName(param.inputs[i].moving);
    readmov->Update();
    ip.moving = readmov->GetOutput();

    // Allocate the gradient
    LDDMMType::alloc_vimg(ip.grad_moving, ip.moving);

    // Precompute the gradient of the moving images. There should be some
    // smoothing of the input images before applying this computation!
    LDDMMType::image_gradient(ip.moving, ip.grad_moving);

    // Set weight
    ip.weight = param.inputs[i].weight;

    // Append
    imgRaw.push_back(ip);
    }
}

template <unsigned int VDim, typename TReal>
void GreedyApproach<VDim, TReal>
::ResampleImages(GreedyParameters &param,
                 const std::vector<GreedyApproach::ImagePair> &imgRaw,
                 std::vector<GreedyApproach::ImagePair> &img, int level)
{
  // The scaling factor
  int shrink_factor = 1 << (param.iter_per_level.size() - (1 + level));

  // What to do at this level?
  if(level == param.iter_per_level.size() - 1)
    {
    img = imgRaw;
    }
  else
    {
    // Resample the input images to lower resolution
    for(int i = 0; i < imgRaw.size(); i++)
      {
      // Create the new image pair
      ImagePair ip;
      ip.moving = ImageType::New();
      ip.fixed = ImageType::New();

      // Resample the images to lower resolution
      // We should smooth the raw image by a nyquist-appropriate kernel first
      // TODO: sigmas - units
      LDDMMType::img_downsample(imgRaw[i].moving, ip.moving, shrink_factor);
      LDDMMType::img_downsample(imgRaw[i].fixed, ip.fixed, shrink_factor);

      // Compute the gradient of the moving image
      LDDMMType::alloc_vimg(ip.grad_moving, ip.moving);

      // Precompute the gradient of the moving images. There should be some
      // smoothing of the input images before applying this computation!
      LDDMMType::image_gradient(ip.moving, ip.grad_moving);

      // Keep the weight the same
      ip.weight = imgRaw[i].weight;

      // Add to the list
      img.push_back(ip);
      }
    }
}

/*
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::RunAffine(GreedyParameters &param)
{
  // Read the image pairs to register
  std::vector<ImagePair> imgRaw;
  ReadImages(param, imgRaw);

  // Create a representation of the current transformation
  typedef vnl_matrix_fixed<double, VDim, VDim> Mat;
  typedef vnl_vector_fixed<double, VDim> Vec;
  Mat A_level; Vec b_level;
  A_level.set_identity();
  b_level.fill(0.0);

  // Initialize the transformation such that the center voxel of the moving image
  // maps to the center voxel of the fixed image
  ImagePair &pair = imgRaw.front();

  // Compute the center index of the moving and fixed images
  itk::ContinuousIndex<double, VDim> ctrFix, ctrMov;
  for(int i = 0; i < VDim; i++)
    {
    ctrFix[i] = pair.fixed->GetBufferedRegion().GetIndex()[i]
                + 0.5 * pair.fixed->GetBufferedRegion().GetSize()[i];
    ctrMov[i] = pair.moving->GetBufferedRegion().GetIndex()[i]
                + 0.5 * pair.moving->GetBufferedRegion().GetSize()[i];
    }

  // Map to the physical location
  itk::Point<double, VDim> pctrFix, pctrMov;
  pair.fixed->TransformContinuousIndexToPhysicalPoint(ctrFix, pctrFix);
  pair.moving->TransformContinuousIndexToPhysicalPoint(ctrMov, pctrMov);

  // Initialize the transform
  for(int i = 0; i < VDim; i++)
    b_level[i] = pctrMov - pctrFix;

  // Iterate over resolution levels
  // The number of resolution levels
  int nlevels = param.iter_per_level.size();

  // Iterate over the resolution levels
  for(unsigned int level = 0; level < nlevels; ++level)
    {
    // Reference space
    ImagePointer refspace = NULL;

    // Intermediate images
    ImagePointer iTemp = ImageType::New();
    VectorImagePointer viTemp = VectorImageType::New();
    VectorImagePointer uk = VectorImageType::New();

    // Prepare the data for this iteration
    std::vector<ImagePair> img;

    // The scaling factor
    int shrink_factor = 1 << (nlevels - (1 + level));

    // Perform resampling
    ResampleImages(param, imgRaw, img, level);

    // Allocate the intermediate data
    refspace = img.front().fixed;
    LDDMMType::alloc_vimg(uk, refspace);
    LDDMMType::alloc_img(iTemp, refspace);
    LDDMMType::alloc_vimg(viTemp, refspace);

    // There is no need to initialize the affine transformation, since it is defined in
    // the physical rather than voxel space

    // Set up the optimization problem

    // Iterate for this level
    for(unsigned int iter = 0; iter < param.iter_per_level[level]; iter++)
      {
      // Initialize u(k+1) to zero
      uk1->FillBuffer(typename LDDMMType::Vec(0.0));

      // Initialize the energy computation
      double total_energy = 0.0;

      // Add all the derivative terms
      for(int j = 0; j < img.size(); j++)
        {
        // Interpolate each moving image
        LDDMMType::interp_img(img[j].moving, uk, iTemp);

        // Dump the moving image?
        if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
          {
          char fname[256];
          sprintf(fname, "dump_moving_%02d_lev%02d_iter%04d.nii.gz", j, level, iter);
          LDDMMType::img_write(iTemp, fname);
          }

        // Subtract the fixed image
        LDDMMType::img_subtract_in_place(iTemp, img[j].fixed);

        // Record the norm of the difference image
        total_energy += img[j].weight * LDDMMType::img_euclidean_norm_sq(iTemp);

        // Interpolate the gradient of the moving image
        LDDMMType::interp_vimg(img[j].grad_moving, uk, 1.0, viTemp);
        LDDMMType::vimg_multiply_in_place(viTemp, iTemp);

        // Accumulate to the force
        LDDMMType::vimg_add_scaled_in_place(uk1, viTemp, -img[j].weight * param.epsilon);
        }

      if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
        {
        char fname[256];
        sprintf(fname, "dump_graduent_iter%04d.nii.gz", iter);
        LDDMMType::vimg_write(uk1, fname);
        }

      // We have now computed the gradient vector field. Next, we smooth it
      LDDMMType::vimg_smooth(uk1, viTemp, param.sigma_pre * shrink_factor);
      // fft.convolution_fft(uk1, kernel, true, viTemp); // 'GradJt0' stores K[ GradJt0 * (det Phi_t1)(Jt1-Jt0) ]

      // Write Uk1
      if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
        {
        char fname[256];
        sprintf(fname, "dump_optflow_lev%02d_iter%04d.nii.gz", level, iter);
        LDDMMType::vimg_write(viTemp, fname);
        }

      // Compute the updated deformation field - in uk1
      LDDMMType::interp_vimg(uk, viTemp, 1.0, uk1);
      LDDMMType::vimg_add_in_place(uk1, viTemp);

      if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
        {
        char fname[256];
        sprintf(fname, "dump_uk1_lev%02d_iter%04d.nii.gz", level, iter);
        LDDMMType::vimg_write(uk1, fname);
        }

      // Swap uk and uk1 pointers
      // VectorImagePointer tmpptr = uk1; uk1 = uk; uk = tmpptr;

      // Another layer of smoothing - really?
      LDDMMType::vimg_smooth(uk1, uk, param.sigma_post * shrink_factor);


      // Report the energy
      // printf("Iter %5d:    Energy = %8.4f     DetJac Range: %8.4f  to %8.4f \n", iter, total_energy, jac_min, jac_max);
      printf("Level %5d,  Iter %5d:    Energy = %8.4f\n", level, iter, total_energy);
      }

    // Store the end result
    uLevel = uk;

    // Compute the jacobian of the deformation field
    LDDMMType::field_jacobian_det(uk, iTemp);
    TReal jac_min, jac_max;
    LDDMMType::img_min_max(iTemp, jac_min, jac_max);
    printf("END OF LEVEL %5d    DetJac Range: %8.4f  to %8.4f \n", level, jac_min, jac_max);

    }

  // Write the resulting transformation field
  LDDMMType::vimg_write(uLevel, param.output.c_str());
}

*/

/**
 * This is the main function of the GreedyApproach algorithm
 */
template <unsigned int VDim, typename TReal>
int GreedyApproach<VDim, TReal>
::Run(GreedyParameters &param)
{
  // Create an optical flow helper object
  OFHelperType of_helper;

  // Set the scaling factors for multi-resolution
  of_helper.SetDefaultPyramidFactors(param.iter_per_level.size());

  // Read the image pairs to register
  ReadImages(param, of_helper);

  // Generate the optimized composite images
  of_helper.BuildCompositeImages();

  // An image pointer desribing the current estimate of the deformation
  VectorImagePointer uLevel = NULL;

  // The number of resolution levels
  int nlevels = param.iter_per_level.size();

  // Iterate over the resolution levels
  for(unsigned int level = 0; level < nlevels; ++level)
    {
    // The scaling factor
    int shrink_factor = 1 << (nlevels - (1 + level));

    // Reference space
    ImageBaseType *refspace = of_helper.GetReferenceSpace(level);

    // Intermediate images
    ImagePointer iTemp = ImageType::New();
    VectorImagePointer viTemp = VectorImageType::New();
    VectorImagePointer uk = VectorImageType::New();
    VectorImagePointer uk1 = VectorImageType::New();

    // Allocate the intermediate data
    LDDMMType::alloc_vimg(uk, refspace);
    LDDMMType::alloc_img(iTemp, refspace);
    LDDMMType::alloc_vimg(viTemp, refspace);
    LDDMMType::alloc_vimg(uk1, refspace);

    // Initialize the deformation field from last iteration
    if(uLevel.IsNotNull())
      {
      LDDMMType::vimg_resample_identity(uLevel, refspace, uk);
      }

    // Iterate for this level
    for(unsigned int iter = 0; iter < param.iter_per_level[level]; iter++)
      {
      // Initialize u(k+1) to zero
      uk1->FillBuffer(typename LDDMMType::Vec(0.0));

      // Compute the gradient of objective
      double total_energy = of_helper.ComputeOpticalFlowField(level, uk, uk1, param.epsilon);

      /*

      // Add all the derivative terms
      for(int j = 0; j < img.size(); j++)
        {
        // Interpolate each moving image
        LDDMMType::interp_img(img[j].moving, uk, iTemp);

        // Dump the moving image?
        if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
          {
          char fname[256];
          sprintf(fname, "dump_moving_%02d_lev%02d_iter%04d.nii.gz", j, level, iter);
          LDDMMType::img_write(iTemp, fname);
          }

        // Subtract the fixed image
        LDDMMType::img_subtract_in_place(iTemp, img[j].fixed);

        // Record the norm of the difference image
        total_energy += img[j].weight * LDDMMType::img_euclidean_norm_sq(iTemp);

        // Interpolate the gradient of the moving image
        LDDMMType::interp_vimg(img[j].grad_moving, uk, 1.0, viTemp);
        LDDMMType::vimg_multiply_in_place(viTemp, iTemp);

        // Accumulate to the force
        LDDMMType::vimg_add_scaled_in_place(uk1, viTemp, -img[j].weight * param.epsilon);
        }

        */

      if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
        {
        char fname[256];
        sprintf(fname, "dump_graduent_iter%04d.nii.gz", iter);
        LDDMMType::vimg_write(uk1, fname);
        }

      // We have now computed the gradient vector field. Next, we smooth it
      LDDMMType::vimg_smooth(uk1, viTemp, param.sigma_pre * shrink_factor);
      // fft.convolution_fft(uk1, kernel, true, viTemp); // 'GradJt0' stores K[ GradJt0 * (det Phi_t1)(Jt1-Jt0) ]

      // Write Uk1
      if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
        {
        char fname[256];
        sprintf(fname, "dump_optflow_lev%02d_iter%04d.nii.gz", level, iter);
        LDDMMType::vimg_write(viTemp, fname);
        }

      // Compute the updated deformation field - in uk1
      LDDMMType::interp_vimg(uk, viTemp, 1.0, uk1);
      LDDMMType::vimg_add_in_place(uk1, viTemp);

      if(param.flag_dump_moving && 0 == iter % param.dump_frequency)
        {
        char fname[256];
        sprintf(fname, "dump_uk1_lev%02d_iter%04d.nii.gz", level, iter);
        LDDMMType::vimg_write(uk1, fname);
        }

      // Swap uk and uk1 pointers
      // VectorImagePointer tmpptr = uk1; uk1 = uk; uk = tmpptr;

      // Another layer of smoothing - really?
      LDDMMType::vimg_smooth(uk1, uk, param.sigma_post * shrink_factor);

      // Compute the Jacobian determinant of the updated field (temporary)
          /*
        */

      // Report the energy
      // printf("Iter %5d:    Energy = %8.4f     DetJac Range: %8.4f  to %8.4f \n", iter, total_energy, jac_min, jac_max);
      printf("Level %5d,  Iter %5d:    Energy = %8.4f\n", level, iter, total_energy);
      }

    // Store the end result
    uLevel = uk;

    // Compute the jacobian of the deformation field
    LDDMMType::field_jacobian_det(uk, iTemp);
    TReal jac_min, jac_max;
    LDDMMType::img_min_max(iTemp, jac_min, jac_max);
    printf("END OF LEVEL %5d    DetJac Range: %8.4f  to %8.4f \n", level, jac_min, jac_max);

    }

  // The transformation field is in voxel units. To work with ANTS, it must be mapped
  // into physical offset units - just scaled by the spacing?

  // Write the resulting transformation field
  VectorImagePointer uPhys = VectorImageType::New();
  LDDMMType::alloc_vimg(uPhys, uLevel);
  LDDMMType::warp_voxel_to_physical(uLevel, uPhys);
  LDDMMType::vimg_write(uPhys, param.output.c_str());

  return 0;
}




int main(int argc, char *argv[])
{
  GreedyParameters param;
  double current_weight = 1.0;

  param.dim = 2;
  param.flag_dump_moving = false;
  param.flag_optimize_affine = false;
  param.dump_frequency = 1;
  param.epsilon = 1.0;
  param.sigma_pre = 3.0;
  param.sigma_post = 1.0;

  param.iter_per_level.push_back(100);
  param.iter_per_level.push_back(100);

  if(argc < 3)
    return usage();

  for(int i = 1; i < argc; ++i)
    {
    std::string arg = argv[i];
    if(arg == "-d")
      {
      param.dim = atoi(argv[++i]);
      }
    else if(arg == "-n")
      {
      std::istringstream f(argv[++i]);
      std::string s;
      param.iter_per_level.clear();
      while (getline(f, s, 'x'))
        param.iter_per_level.push_back(atoi(s.c_str()));
      }
    else if(arg == "-w")
      {
      current_weight = atof(argv[++i]);
      }
    else if(arg == "-e")
      {
      param.epsilon = atof(argv[++i]);
      }
    else if(arg == "-s")
      {
      param.sigma_pre = atof(argv[++i]);
      param.sigma_post = atof(argv[++i]);
      }
    else if(arg == "-i")
      {
      ImagePairSpec ip;
      ip.weight = current_weight;
      ip.fixed = argv[++i];
      ip.moving = argv[++i];
      param.inputs.push_back(ip);
      }
    else if(arg == "-o")
      {
      param.output = argv[++i];
      }
    else if(arg == "-dump-moving")
      {
      param.flag_dump_moving = true;
      }
    else if(arg == "-dump-frequency")
      {
      param.dump_frequency = atoi(argv[++i]);
      }
    else if(arg == "-a")
      {
      param.flag_optimize_affine = true;
      }
    else
      {
      std::cerr << "Unknown parameter " << arg << std::endl;
      return -1;
      }
    }

  if(param.flag_optimize_affine)
    {
    switch(param.dim)
      {/*
      case 2: return GreedyApproach<2>::RunAffine(param); break;
      case 3: return GreedyApproach<3>::RunAffine(param); break;
      case 4: return GreedyApproach<4>::RunAffine(param); break;
      default:
            std::cerr << "Wrong dimensionality" << std::endl;
            return -1;
      */}
    }
  else
    {
    switch(param.dim)
      {
      case 2: return GreedyApproach<2>::Run(param); break;
      case 3: return GreedyApproach<3>::Run(param); break;
      case 4: return GreedyApproach<4>::Run(param); break;
      default:
            std::cerr << "Wrong dimensionality" << std::endl;
            return -1;
      }
    }
}
