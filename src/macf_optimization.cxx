#include "MultiImageRegistrationHelper.h"
#include <string>
#include <fstream>
#include <vector>

#include "lddmm_data.h"

#include "CommandLineHelper.h"

using namespace std;

struct MACFParameters
{
  string fnReference, fnPsiPattern, fnWeightPattern, fnIds, fnOutPhiPattern;
  string fnGrayPattern, fnOutIterTemplatePattern;

  int exponent;
  double sigma1, sigma2;
  double epsilon;
  int n_iter;


  MACFParameters()
    {
    exponent = 6;
    sigma1 = sqrt(3.0);
    sigma2 = sqrt(0.5);
    epsilon = 0.25;
    n_iter = 100;
    }
};

int usage()
{
  MACFParameters p;

  printf("macf_optimize: optimization routine for CVPR 2012 MACF paper\n");
  printf("usage:\n");
  printf("  macf_optimize [options]\n");
  printf("required options:\n");
  printf("  -d <2|3>             : number of dimensions\n");
  printf("  -ids <file>          : text file of ids\n");
  printf("  -ref <image>         : reference image\n");
  printf("  -psi <pattern_2s>    : pattern of root warps of psi\n");
  printf("  -wgt <pattern_2s>    : pattern of weight images\n");
  printf("  -o <pattern_1s>      : output phi pattern\n");
  printf("additional options:\n");
  printf("  -n <value>           : number of iterations (def = %d)\n",p.n_iter);
  printf("  -eps <value>         : step size (def = %f)\n",p.epsilon);
  printf("  -exp <value>         : exponent for scaling and squaring (def = %d)\n",p.exponent);
  printf("  -s <sigma1> <sigma2> : smoothing in voxels (def = %f, %f)\n",p.sigma1,p.sigma2);
  printf("  -img <pattern_1s>    : pattern of grayscale images (for visualizing registration)\n");
  printf("  -otemp <pattern_1d>  : pattern for saving templates at each iteration\n");
  printf("patterns:\n");
  printf("  pattern_1s           : of form blah_%%s_blah.nii.gz\n");
  printf("  pattern_2s           : of form blah_%%s_blah_%%s_blah.nii.gz (fixed, then moving)\n");
  printf("  pattern_1d           : of form blah_%%03d_blah.nii.gz\n");
  return -1;
}

template <typename TFloat, unsigned int VDim>
class MACFWorker
{
public:

  typedef LDDMMData<TFloat, VDim> LDDMMType;
  typedef MultiImageOpticalFlowHelper<TFloat, VDim> OFHelperType;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::ImageType ImageType;
  typedef typename LDDMMType::MatrixImageType MatrixImageType;
  typedef typename VectorImageType::Pointer VectorImagePointer;
  typedef typename ImageType::Pointer ImagePointer;
  typedef typename MatrixImageType::Pointer MatrixImagePointer;

  void ReadImages()
    {
    // Buffer for expanding printf-like patterns
    char fn[1024];

    // Read the list of ids
    ifstream iff(m_Param.fnIds);
    string id;
    while(iff >> id)
      m_Ids.push_back(id);

    // Allocate the main storage
    m_Size = m_Ids.size();
    m_Data.resize(m_Size);

    // Read the reference image
    m_Reference = LDDMMType::img_read(m_Param.fnReference.c_str());

    // Some working images
    m_Work = LDDMMType::alloc_vimg(m_Reference);
    m_ScalarWork = LDDMMType::alloc_img(m_Reference);

    // Jacobian storage
    MatrixImagePointer jac = LDDMMType::alloc_mimg(m_Reference);
    MatrixImagePointer jac_work = LDDMMType::alloc_mimg(m_Reference);

    // Create all the pair data
    for(int i = 0; i < m_Size; i++)
      {
      m_Data[i].u = LDDMMType::alloc_vimg(m_Reference);
      m_Data[i].u_root = LDDMMType::alloc_vimg(m_Reference);
      m_Data[i].grad_u = LDDMMType::alloc_vimg(m_Reference);
      m_Data[i].delta = LDDMMType::alloc_vimg(m_Reference);
      
      m_Data[i].pair_data.resize(m_Size);

      for(int j = 0; j < m_Size; j++)
        {
        if(i != j)
          {
          // Reference the current pair data
          PairData &pd = m_Data[i].pair_data[j];

          // Read the psi root image
          sprintf(fn, m_Param.fnPsiPattern.c_str(), m_Ids[i].c_str(), m_Ids[j].c_str());
          VectorImagePointer psi_root = LDDMMType::vimg_read(fn);
          OFHelperType::PhysicalWarpToVoxelWarp(psi_root, psi_root, psi_root);

          // Integrate the psi image forward
          pd.psi_forward = LDDMMType::alloc_vimg(m_Reference);
          LDDMMType::vimg_exp(psi_root, pd.psi_forward, m_Work, m_Param.exponent, 1.0);

          // Integrate the psi image backward with jacobian 
          pd.psi_inverse = LDDMMType::alloc_vimg(m_Reference);
          LDDMMType::vimg_exp_with_jacobian(
            psi_root, pd.psi_inverse, m_Work, jac, jac_work, m_Param.exponent, -1.0);

          // Compute Jacobian determinant
          LDDMMType::mimg_det(jac, 1.0, m_ScalarWork);

          // Load the weight image
          sprintf(fn, m_Param.fnWeightPattern.c_str(), m_Ids[i].c_str(), m_Ids[j].c_str());
          pd.wgt_fixed = LDDMMType::img_read(fn);

          // Warp the weight by the inverse psi and scale by the determinant
          pd.wgt_moving = LDDMMType::alloc_img(m_Reference);
          LDDMMType::interp_img( pd.wgt_fixed, pd.psi_inverse, pd.wgt_moving, false, false, 0);
          LDDMMType::img_multiply_in_place(pd.wgt_moving, m_ScalarWork);

          cout << "." << flush;
          }
        }

      // Read the optional grayscale images
      if(m_Param.fnGrayPattern.size())
        {
        sprintf(fn, m_Param.fnGrayPattern.c_str(), m_Ids[i].c_str());
        m_Data[i].img_gray = LDDMMType::img_read(fn);
        }

      cout << "." << endl;
      }
    }

  double ComputeDeltasAndObjective()
    {
    double total_error = 0;

    // Compute the deltas and the objective
    for(int i = 0; i < m_Size; i++)
      {
      // Set the delta to the current u_i
      LDDMMType::vimg_copy(m_Data[i].u, m_Data[i].delta);

      // Add all the differences
      for(int j = 0; j < m_Size; j++)
        {
        if(j != i)
          {
          PairData &pd = m_Data[i].pair_data[j];
          LDDMMType::interp_vimg(m_Data[j].u, pd.psi_forward, 1.0, m_Work);
          LDDMMType::vimg_add_in_place(m_Work, pd.psi_forward);
          LDDMMType::vimg_multiply_in_place(m_Work, pd.wgt_fixed);
          LDDMMType::vimg_subtract_in_place(m_Data[i].delta, m_Work);
          }
        }

      // Compute the norm of the delta
      m_Data[i].norm_delta = LDDMMType::vimg_euclidean_norm_sq(m_Data[i].delta);

      // Add to the total error
      total_error += m_Data[i].norm_delta;
      }

    // Extract the average error per pixel per image
    total_error /= m_Size * m_Reference->GetBufferedRegion().GetNumberOfPixels();

    return total_error;
    }

  void BuildTemplate(int iter)
    {
    // We need a couple of images
    ImagePointer templ = LDDMMType::alloc_img(m_Reference);
    VectorImagePointer phi_exp = LDDMMType::alloc_vimg(m_Reference);

    // Iterate over the images
    for(int i = 0; i < m_Size; i++)
      {
      // Compute the deformation that warps i-th image into template space
      LDDMMType::vimg_exp(m_Data[i].u_root, phi_exp, m_Work, m_Param.exponent, -1.0);

      // Apply that warp to the gray image
      LDDMMType::interp_img(m_Data[i].img_gray, phi_exp, m_ScalarWork, false, false, 0);

      // Add the image to the template
      LDDMMType::img_add_in_place(templ, m_ScalarWork);
      }

    // Scale the template by the number of images
    LDDMMType::img_scale_in_place(templ, 1.0 / m_Size);

    // Write the template
    char fn[1024];
    sprintf(fn, m_Param.fnOutIterTemplatePattern.c_str(), iter);
    LDDMMType::img_write(templ, fn);
    }

  void ComputeGradientAndUpdate()
    {
    double global_max_norm = 0.0;

    // Compute gradients and their norms
    for(int m = 0; m < m_Size; m++)
      {
      // Start by adding the delta
      LDDMMType::vimg_copy(m_Data[m].delta, m_Data[m].grad_u);

      // Subtract each of the deltas warped into moving space
      for(int j = 0; j < m_Size; j++)
        {
        if(m != j)
          {
          PairData &pd = m_Data[j].pair_data[m];
          LDDMMType::interp_vimg(m_Data[j].delta, pd.psi_inverse, 1.0, m_Work);
          LDDMMType::vimg_multiply_in_place(m_Work, pd.wgt_moving);
          LDDMMType::vimg_subtract_in_place(m_Data[m].grad_u, m_Work);
          }
        }

      // Smooth the gradient 
      LDDMMType::vimg_smooth_withborder(m_Data[m].grad_u, m_Work, m_Param.sigma1, 1);

      // Compute the norm of the gradient
      TFloat norm_min, norm_max;
      LDDMMType::vimg_norm_min_max(m_Data[m].grad_u, m_ScalarWork, norm_min, norm_max);
      if(norm_max > global_max_norm)
        global_max_norm = norm_max;
      }

    // Compute the scaling factor
    double scale = 1.0 / (2 << m_Param.exponent);
    if(global_max_norm > m_Param.epsilon)
      scale = scale * m_Param.epsilon / global_max_norm;

    // Scale everything down by the max norm and smooth again
    for(int m = 0; m < m_Size; m++)
      {
      // Compute the updated root warp
      LDDMMType::vimg_copy(m_Data[m].u_root, m_Work);
      LDDMMType::vimg_add_scaled_in_place(m_Work, m_Data[m].grad_u, -scale);
      LDDMMType::vimg_smooth_withborder(m_Work, m_Data[m].u_root, m_Param.sigma2, 1);

      // Exponentiate the root warps
      LDDMMType::vimg_exp(m_Data[m].u_root, m_Data[m].u, m_Work, m_Param.exponent, 1.0);
      }
    }

  void WriteResults()
    {
    for(int i = 0; i < m_Size; i++)
      {
      // Map the warp back into physical units
      OFHelperType::VoxelWarpToPhysicalWarp(m_Data[i].u_root, m_Reference, m_Work);
      char fn[1024];
      sprintf(fn, m_Param.fnOutPhiPattern.c_str(), m_Ids[i].c_str());
      LDDMMType::vimg_write(m_Work, fn); 
      }
    }

  void Run()
    {
    // Read the images into the datastructure
    ReadImages();
    printf("Read images for %d ids\n", m_Size);

    // Iterate
    for(int iter = 0; iter < m_Param.n_iter; iter++)
      {
      // Compute the objective and deltas
      double total_error = ComputeDeltasAndObjective();
      printf("Iter %04d:   Total Error: %12.4f\n", iter, total_error);

      // Write the iteration template
      if(m_Param.fnGrayPattern.size() && m_Param.fnOutIterTemplatePattern.size())
        BuildTemplate(iter);

      // Compute the gradients 
      ComputeGradientAndUpdate();
      }

    // Write out final warps
    WriteResults();
    }

  MACFWorker(const MACFParameters &param) 
    : m_Param(param) {}

protected:

  struct PairData 
    {
    VectorImagePointer psi_forward, psi_inverse;
    ImagePointer wgt_fixed, wgt_moving;
    };

  struct ImageData
    {
    vector<PairData> pair_data;
    VectorImagePointer u, u_root, grad_u;
    VectorImagePointer delta;
    ImagePointer img_gray;
    double norm_delta;
    };

  VectorImagePointer m_Work;
  ImagePointer m_ScalarWork;
  MACFParameters m_Param;

  ImagePointer m_Reference;


  vector<ImageData> m_Data;
  vector<string> m_Ids;
  int m_Size;

};

int main(int argc, char *argv[])
{
  MACFParameters param;
  int dim = 2;

  if(argc < 2)
    return usage();

  CommandLineHelper cl(argc, argv);
  while(!cl.is_at_end())
    {
    // Read the next command
    std::string arg = cl.read_command();

    if(arg == "-d")
      {
      dim = cl.read_integer();
      }
    else if(arg == "-ids")
      {
      param.fnIds = cl.read_existing_filename();
      }
    else if(arg == "-ref")
      {
      param.fnReference = cl.read_existing_filename();
      }
    else if(arg == "-psi")
      {
      param.fnPsiPattern = cl.read_string();
      }
    else if(arg == "-wgt")
      {
      param.fnWeightPattern = cl.read_string();
      }
    else if(arg == "-o")
      {
      param.fnOutPhiPattern = cl.read_string();
      }
    else if(arg == "-img")
      {
      param.fnGrayPattern = cl.read_string();
      }
    else if(arg == "-exp")
      {
      param.exponent = cl.read_integer();
      }
    else if(arg == "-eps")
      {
      param.epsilon = cl.read_double();
      }
    else if(arg == "-s")
      {
      param.sigma1 = cl.read_double();
      param.sigma2 = cl.read_double();
      }
    else if(arg == "-n")
      {
      param.n_iter = cl.read_integer();
      }
    else if(arg == "-otemp")
      {
      param.fnOutIterTemplatePattern = cl.read_string();
      }
    else
      {
      printf("Unknown parameter %s\n", arg.c_str());
      return -1;
      }
    }

  if(dim == 2)
    {
    MACFWorker<float, 2> worker(param);
    worker.Run();
    }
  else if(dim == 3)
    {
    MACFWorker<float, 3> worker(param);
    worker.Run();
    }
}
