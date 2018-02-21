#include "MultiImageRegistrationHelper.h"
#include <string>
#include <fstream>
#include <vector>

#include "lddmm_data.h"

#include "CommandLineHelper.h"

using namespace std;

int usage()
{
  printf("macf_optimize: optimization routine for CVPR 2012 MACF paper\n");
  printf("usage:\n");
  printf("  macf_optimize [options]\n");
  printf("options:\n");
  printf("  -d <2|3>         : number of dimensions\n");
  printf("  -ids <file>      : text file of ids\n");
  printf("  -ref <image>     : refernce image\n");
  printf("  -psi <pattern>   : pattern of root warps of psi\n");
  printf("  -wgt <pattern>   : pattern of weight images\n");
  printf("  -o <pattern>     : output phi pattern\n");
  return -1;
}

struct MACFParameters
{
  string fnReference, fnPsiPattern, fnWeightPattern, fnIds, fnOutPhiPattern;

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

template <typename TFloat, unsigned int VDim>
class MACFWorker
{
public:

  typedef LDDMMData<TFloat, VDim> LDDMMType;
  typedef MultiImageOpticalFlowHelper<TFloat, VDim> OFHelperType;
  typedef typename LDDMMType::VectorImageType VectorImageType;
  typedef typename LDDMMType::ImageType ImageType;
  typedef typename VectorImageType::Pointer VectorImagePointer;
  typedef typename ImageType::Pointer ImagePointer;

  void ReadImages()
    {
    // Read the list of ids
    ifstream iff(m_Param.fnIds);
    string id;
    while(iff >> id)
      m_Ids.push_back(id);

    // Allocate the main storage
    int m_Size = m_Ids.size();
    m_Data.resize(m_Size);

    // Read the reference image
    ImagePointer refimg = ImageType::New();
    LDDMMType::img_read(m_Param.fnReference.c_str(), refimg);

    // Some working images
    m_Work = VectorImageType::New();
    LDDMMType::alloc_vimg(m_Work, refimg);

    m_WorkNorm = ImageType::New();
    LDDMMType::alloc_img(m_WorkNorm, refimg);

    // Create all the pair data
    for(int i = 0; i < m_Size; i++)
      {
      m_Data[i].u = VectorImageType::New();
      LDDMMType::alloc_vimg(m_Data[i].u, refimg);

      m_Data[i].u_root = VectorImageType::New();
      LDDMMType::alloc_vimg(m_Data[i].u_root, refimg);

      m_Data[i].grad_u = VectorImageType::New();
      LDDMMType::alloc_vimg(m_Data[i].grad_u, refimg);

      m_Data[i].delta = VectorImageType::New();
      LDDMMType::alloc_vimg(m_Data[i].delta, refimg);
      
      m_Data[i].pair_data.resize(m_Size);

      for(int j = 0; j < m_Size; j++)
        {
        if(i != j)
          {
          // Reference the current pair data
          PairData &pd = m_Data[i].pair_data[j];

          // Read the psi root image
          char fn[1024];
          sprintf(fn, m_Param.fnPsiPattern.c_str(), m_Ids[i].c_str(), m_Ids[j].c_str());
          VectorImagePointer psi_root = VectorImageType::New();
          LDDMMType::vimg_read(fn, psi_root);
          OFHelperType::PhysicalWarpToVoxelWarp(psi_root, psi_root, psi_root);

          // Integrate the psi image forward
          VectorImagePointer psi_exp = VectorImageType::New();
          LDDMMType::alloc_vimg(psi_exp, refimg);
          LDDMMType::vimg_copy(psi_root, psi_exp);
          for(int i = 0; i < m_Param.exponent; i++)
            {
            LDDMMType::interp_vimg(psi_exp, psi_exp, 1.0, m_Work);
            LDDMMType::vimg_add_in_place(psi_exp, m_Work);
            }
          pd.psi_forward = psi_exp;

          // Jacobian matrices
          typedef typename LDDMMType::MatrixImageType JacobianImageType;
          typename JacobianImageType::Pointer jac = JacobianImageType::New();
          LDDMMType::alloc_mimg(jac, refimg);
          typename JacobianImageType::Pointer jac_work = JacobianImageType::New();
          LDDMMType::alloc_mimg(jac_work, refimg);

          // Integrate the psi image backward with jacobian 
          LDDMMType::vimg_copy(psi_root, psi_exp);
          LDDMMType::vimg_scale_in_place(psi_exp, -1.0);
          LDDMMType::field_jacobian(psi_exp, jac);
          for(int i = 0; i < m_Param.exponent; i++)
            {
            // Compute the composition of the Jacobian with itself
            LDDMMType::jacobian_of_composition(jac, jac, psi_exp, jac_work);

            // Swap the pointers, so jac points to the actual composed jacobian
            typename JacobianImageType::Pointer temp = jac_work.GetPointer();
            jac_work = jac.GetPointer();
            jac = temp.GetPointer();

            // Exponentiate the warp itself
            LDDMMType::interp_vimg(psi_exp, psi_exp, 1.0, m_Work);
            LDDMMType::vimg_add_in_place(psi_exp, m_Work);
            }
          pd.psi_inverse = psi_exp;

          // Compute Jacobian determinant
          ImagePointer jac_det = ImageType::New();
          LDDMMType::alloc_img(jac_det, refimg);
          LDDMMType::mimg_det(jac, 1.0, jac_det);

          // Load the weight image
          sprintf(fn, m_Param.fnWeightPattern.c_str(), m_Ids[i].c_str(), m_Ids[j].c_str());
          pd.wgt_fixed = ImageType::New();
          LDDMMType::img_read(fn, pd.wgt_fixed);

          // Warp the weight by the inverse psi and scale by the determinant
          LDDMMType::interp_img( pd.wgt_fixed, psi_exp, pd.wgt_moving, false, false, 0);
          LDDMMType::img_multiply_in_place(pd.wgt_moving, jac_det);
          }
        }
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

    return total_error;
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
          PairData &pd = m_Data[m].pair_data[j];

          LDDMMType::interp_vimg(m_Data[j].delta, pd.psi_inverse, 1.0, m_Work);
          LDDMMType::vimg_multiply_in_place(m_Work, pd.wgt_moving);
          LDDMMType::vimg_subtract_in_place(m_Data[m].grad_u, m_Work);
          }
        }

      // Smooth the gradient 
      LDDMMType::vimg_smooth_withborder(m_Data[m].grad_u, m_Work, m_Param.sigma1, 1);

      // Compute the norm of the gradient
      TFloat norm_min, norm_max;
      LDDMMType::vimg_norm_min_max(m_Data[m].grad_u, m_WorkNorm, norm_min, norm_max);
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
      LDDMMType::vimg_copy(m_Data[m].u_root, m_Data[m].u);
      for(int j = 0; j < m_Param.exponent; j++)
        {
        LDDMMType::interp_vimg(m_Data[m].u, m_Data[m].u, 1.0, m_Work);
        LDDMMType::vimg_add_in_place(m_Data[m].u, m_Work);
        }
      }
    }

  void WriteResults()
    {
    for(int i = 0; i < m_Size; i++)
      {
      char fn[1024];
      sprintf(fn, m_Param.fnOutPhiPattern.c_str(), m_Ids[i].c_str());
      LDDMMType::vimg_write(m_Data[i].u_root, fn); 
      }
    }

  void Run()
    {
    // Read the images into the datastructure
    ReadImages();

    // Iterate
    for(int iter = 0; iter < m_Param.n_iter; iter++)
      {
      // Compute the objective and deltas
      double total_error = ComputeDeltasAndObjective();
      printf("Iter %04d:   Total Error: %12.4f\n", iter, total_error);

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
    double norm_delta;
    };

  VectorImagePointer m_Work;
  ImagePointer m_WorkNorm;
  MACFParameters m_Param;


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
