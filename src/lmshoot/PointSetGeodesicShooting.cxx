#include "PointSetHamiltonianSystem.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_cost_function.h"
#include "vnl/algo/vnl_lbfgs.h"
#include "vnl/vnl_cross.h"

#include <iostream>
#include <algorithm>
#include "GreedyMeshIO.h"
#include "VTKArrays.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"
#include "vtkDoubleArray.h"
#include "vtkQuadricClustering.h"
#include "vtkSmartPointer.h"
#include "vtkCell.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"

#include "PointSetGeodesicShooting.h"
#include "CommandLineHelper.h"
#include "GreedyAPI.h"
#include "itkMultiThreaderBase.h"

using namespace std;

int lmshoot_usage(bool print_template_params)
{
  cout << "lmshoot: Geodesic shooting for landmarks" << endl;
  cout << "Usage:" << endl;
  cout << "  lmshoot [options]" << endl;
  cout << "Required Options:" << endl;
  cout << "  -m template.vtk target.vtk : input meshes" << endl;
  cout << "  -o result.vtk              : output mesh (template with initial momentum)" << endl;
  cout << "  -s sigma                   : LDDMM kernel standard deviation" << endl;
  cout << "Additional Options:" << endl;
  if(print_template_params)
    cout << "  -d dim                     : problem dimension (3)" << endl;
  cout << "  -G                         : Compute global similarity transform, not geodesic shooting" << endl;
  cout << "  -n N                       : number of time steps (100)" << endl;
  cout << "  -R                         : use Ralston integration instead of the default Euler method" << endl;
  cout << "  -a <L|C|V>                 : data attachment term, L for landmark euclidean distance (default), " << endl;
  cout << "                               C for current metric, V for varifold metric." << endl;
  cout << "  -l lambda                  : weight of the data attachment term (1.0)" << endl;
  cout << "  -g gamma                   : weight of the Hamiltonian regularization term (1.0)" << endl;
  cout << "  -S sigma                   : kernel standard deviation for current/varifold metric" << endl;
  cout << "  -c mesh.vtk                : optional control point mesh (if different from template.vtk)" << endl;
  cout << "  -p array_name              : read initial momentum from named array in control/template mesh" << endl;
  cout << "  -i iter_grad iter_newt     : max iterations for optimization for gradient descent and newton's" << endl;
  cout << "  -O filepattern             : pattern for saving traced landmark paths (e.g., path%04d.vtk)" << endl;
  if(print_template_params)
    cout << "  -f                         : use single-precision float (off by deflt)" << endl;
  cout << "  -C mu0 mu_mult             : test constrained optimization (not for general use)" << endl;
  cout << "  -t n_threads               : limit number of concurrent threads to n_threads" << endl;
  cout << "  -D n                       : perform derivative check (for first n momenta)" << endl;
  cout << "  -L array_name              : use label-restricted data attachment, with label posteriors in given array" << endl;
  cout << "  -J weight                  : use Jacobian regularization with provided weight (default: no)" << endl;
  return -1;
}

template <class TFloat, unsigned int VDim>
class TriangleCentersAndNormals
{
public:
  typedef PointSetHamiltonianSystem<TFloat, VDim> HSystem;
  typedef typename HSystem::Vector Vector;
  typedef typename HSystem::Matrix Matrix;
  typedef vnl_matrix<int> Triangulation;

  TriangleCentersAndNormals(const Triangulation &tri, bool normalize);
  void Forward(const Matrix &q);
  void Backward(const Matrix &dE_dC, const Matrix &dE_dN, const Vector &dE_dW_norm, Matrix &dE_dq);
};

template <class TFloat>
class TriangleCentersAndNormals<TFloat, 3>
{
public:
  typedef PointSetHamiltonianSystem<TFloat, 3> HSystem;
  typedef typename HSystem::Vector Vector;
  typedef typename HSystem::Matrix Matrix;
  typedef vnl_matrix<int> Triangulation;

  TriangleCentersAndNormals( const Triangulation &tri, bool normalize)
  {
    this->normalize = normalize;
    this->tri = tri;
    this->C.set_size(tri.rows(), 3);
    this->N.set_size(tri.rows(), 3);
    this->U.set_size(tri.rows(), 3);
    this->V.set_size(tri.rows(), 3);
    this->W.set_size(tri.rows(), 3);
    this->W_norm.set_size(tri.rows());
  }

  void Forward(const Matrix &q)
  {
    for(unsigned int i = 0; i < tri.rows(); i++)
      {
      // Get pointer access to the outputs
      TFloat *Ui = U.data_array()[i];
      TFloat *Vi = V.data_array()[i];
      TFloat *Wi = W.data_array()[i];
      TFloat *Ci = C.data_array()[i];
      TFloat *Ni = N.data_array()[i];

      int v0 = tri(i, 0), v1 = tri(i, 1), v2 = tri(i, 2);
      for(unsigned int a = 0; a < 3; a++)
        {
        Ci[a] = (q(v0, a) + q(v1, a) + q(v2, a)) / 3.0;
        Ui[a] = q(v1, a) - q(v0, a);
        Vi[a] = q(v2, a) - q(v0, a);
        }

      if(normalize)
        {
        Wi[0] = 0.5 * (Ui[1] * Vi[2] - Ui[2] * Vi[1]);
        Wi[1] = 0.5 * (Ui[2] * Vi[0] - Ui[0] * Vi[2]);
        Wi[2] = 0.5 * (Ui[0] * Vi[1] - Ui[1] * Vi[0]);

        // Compute the norm of the cross-product
        W_norm[i] = sqrt(Wi[0] * Wi[0] + Wi[1] * Wi[1] + Wi[2] * Wi[2]);
        if(W_norm[i] > 0.0)
          {
          Ni[0] = Wi[0] / W_norm[i];
          Ni[1] = Wi[1] / W_norm[i];
          Ni[2] = Wi[2] / W_norm[i];
          }
        else
          {
          Ni[0] = 0.0;
          Ni[1] = 0.0;
          Ni[2] = 0.0;
          }
        }
      else
        {
        // Compute the cross-product and store in the normal (this is what currents use)
        Ni[0] = 0.5 * (Ui[1] * Vi[2] - Ui[2] * Vi[1]);
        Ni[1] = 0.5 * (Ui[2] * Vi[0] - Ui[0] * Vi[2]);
        Ni[2] = 0.5 * (Ui[0] * Vi[1] - Ui[1] * Vi[0]);
        }
      }
  }

  void Backward(const Matrix &dE_dC, const Matrix &dE_dN, const Vector &dE_dW_norm, Matrix &dE_dq)
  {
    dE_dq.fill(0.0);
    TFloat dU[3], dV[3], dW[3];

    for(unsigned int i = 0; i < tri.rows(); i++)
      {
      // Get pointer access to the outputs
      TFloat *Ui = U.data_array()[i];
      TFloat *Vi = V.data_array()[i];
      TFloat *Wi = W.data_array()[i];
      TFloat *Ni = N.data_array()[i];
      const TFloat *dCi = dE_dC.data_array()[i];
      const TFloat *dNi = dE_dN.data_array()[i];
      TFloat dW_norm = dE_dW_norm[i];

      // Get the vertex indices and the corresponding gradients
      int v0 = tri(i, 0), v1 = tri(i, 1), v2 = tri(i, 2);
      TFloat *dq0 = dE_dq.data_array()[v0];
      TFloat *dq1 = dE_dq.data_array()[v1];
      TFloat *dq2 = dE_dq.data_array()[v2];

      if(normalize)
        {
        // Partial of the norm of W
        if(W_norm[i] > 0.0)
          {
          dW[0] = ((1 - Ni[0]*Ni[0]) * dNi[0] - Ni[0] * Ni[1] * dNi[1] - Ni[0] * Ni[2] * dNi[2] + Wi[0] * dW_norm) / W_norm[i];
          dW[1] = ((1 - Ni[1]*Ni[1]) * dNi[1] - Ni[1] * Ni[0] * dNi[0] - Ni[1] * Ni[2] * dNi[2] + Wi[1] * dW_norm) / W_norm[i];
          dW[2] = ((1 - Ni[2]*Ni[2]) * dNi[2] - Ni[2] * Ni[0] * dNi[0] - Ni[2] * Ni[1] * dNi[1] + Wi[2] * dW_norm) / W_norm[i];
          }
        else
          {
          dW[0] = dNi[0];
          dW[1] = dNi[1];
          dW[2] = dNi[2];
          }

        // Backprop the cross-product
        dU[0] = 0.5 * (Vi[1] * dW[2] - Vi[2] * dW[1]);
        dU[1] = 0.5 * (Vi[2] * dW[0] - Vi[0] * dW[2]);
        dU[2] = 0.5 * (Vi[0] * dW[1] - Vi[1] * dW[0]);

        dV[0] = 0.5 * (Ui[2] * dW[1] - Ui[1] * dW[2]);
        dV[1] = 0.5 * (Ui[0] * dW[2] - Ui[2] * dW[0]);
        dV[2] = 0.5 * (Ui[1] * dW[0] - Ui[0] * dW[1]);
        }
      else
        {
        // Backprop the cross-product
        dU[0] = 0.5 * (Vi[1] * dNi[2] - Vi[2] * dNi[1]);
        dU[1] = 0.5 * (Vi[2] * dNi[0] - Vi[0] * dNi[2]);
        dU[2] = 0.5 * (Vi[0] * dNi[1] - Vi[1] * dNi[0]);

        dV[0] = 0.5 * (Ui[2] * dNi[1] - Ui[1] * dNi[2]);
        dV[1] = 0.5 * (Ui[0] * dNi[2] - Ui[2] * dNi[0]);
        dV[2] = 0.5 * (Ui[1] * dNi[0] - Ui[0] * dNi[1]);
        }

      // Backprop the Ui and Vi
      for(unsigned int a = 0; a < 3; a++)
        {
        // The center contribution
        TFloat dCi_a  = dCi[a] / 3.0;
        dq0[a] += dCi_a - dU[a] - dV[a];
        dq1[a] += dCi_a + dU[a];
        dq2[a] += dCi_a + dV[a];
        }
      }
  }

  // Whether or not we normalize the values, normalization
  // is needed for the Varifold metric but not for Currents
  bool normalize;

  // Intermediate values
  Triangulation tri;
  Matrix U, V, W;
  Vector W_norm;

  // Results
  Matrix C, N;
};


template <class TFloat>
class TriangleCentersAndNormals<TFloat, 2>
{
public:
  typedef PointSetHamiltonianSystem<TFloat, 2> HSystem;
  typedef typename HSystem::Vector Vector;
  typedef typename HSystem::Matrix Matrix;
  typedef vnl_matrix<int> Triangulation;

  TriangleCentersAndNormals( const Triangulation &tri, bool normalize)
  {
    this->normalize = normalize;
    this->tri = tri;
    this->C.set_size(tri.rows(), 2);
    this->N.set_size(tri.rows(), 2);
    this->U.set_size(tri.rows(), 2);
    this->W.set_size(tri.rows(), 2);
    this->W_norm.set_size(tri.rows());
  }

  void Forward(const Matrix &q)
  {
    for(unsigned int i = 0; i < tri.rows(); i++)
      {
      // Get pointer access to the outputs
      TFloat *Ui = U.data_array()[i];
      TFloat *Wi = W.data_array()[i];
      TFloat *Ci = C.data_array()[i];
      TFloat *Ni = N.data_array()[i];

      int v0 = tri(i, 0), v1 = tri(i, 1);
      for(unsigned int a = 0; a < 2; a++)
        {
        Ci[a] = (q(v0, a) + q(v1, a)) / 2.0;
        Ui[a] = q(v1, a) - q(v0, a);
        }

      if(normalize)
        {
        // Compute the cross-product
        Wi[0] = Ui[1];
        Wi[1] = -Ui[0];

        // Compute the norm of the cross-product
        W_norm[i] = sqrt(Wi[0] * Wi[0] + Wi[1] * Wi[1]);
        if(W_norm[i] > 0.0)
          {
          Ni[0] = Wi[0] / W_norm[i];
          Ni[1] = Wi[1] / W_norm[i];
          }
        else
          {
          Ni[0] = 0.0;
          Ni[1] = 0.0;
          }
        }
      else
        {
        // Compute the cross-product
        Ni[0] = Ui[1];
        Ni[1] = -Ui[0];
        }
      }
  }

  void Backward(const Matrix &dE_dC, const Matrix &dE_dN, const Vector &dE_dW_norm, Matrix &dE_dq)
  {
    dE_dq.fill(0.0);
    TFloat dU[2], dW[2];

    for(unsigned int i = 0; i < tri.rows(); i++)
      {
      // Get pointer access to the outputs
      TFloat *Ni = N.data_array()[i];
      const TFloat *dCi = dE_dC.data_array()[i];
      const TFloat *dNi = dE_dN.data_array()[i];
      TFloat *Wi = W.data_array()[i];
      TFloat dW_norm = dE_dW_norm[i];

      // Get the vertex indices and the corresponding gradients
      int v0 = tri(i, 0), v1 = tri(i, 1);
      TFloat *dq0 = dE_dq.data_array()[v0];
      TFloat *dq1 = dE_dq.data_array()[v1];

      if(normalize)
      {
        // Partial of the norm of W
        if(W_norm[i] > 0.0)
          {
          dW[0] = ((1 - Ni[0]*Ni[0]) * dNi[0] - Ni[0] * Ni[1] * dNi[1] + Wi[0] * dW_norm) / W_norm[i];
          dW[1] = ((1 - Ni[1]*Ni[1]) * dNi[1] - Ni[1] * Ni[0] * dNi[0] + Wi[1] * dW_norm) / W_norm[i];
          }
        else
          {
          dW[0] = dNi[0];
          dW[1] = dNi[1];
          }

        // Backprop the cross-product
        dU[0] = -dW[1];
        dU[1] = dW[0];
        }
      else
        {
        // Backprop the cross-product
        dU[0] = -dNi[1];
        dU[1] = dNi[0];
        }

      // Backprop the Ui and Vi
      for(unsigned int a = 0; a < 2; a++)
        {
        // The center contribution
        TFloat dCi_a  = dCi[a] / 2.0;
        dq0[a] += dCi_a - dU[a];
        dq1[a] += dCi_a + dU[a];
        }
      }
  }

  // Intermediate values
  bool normalize;
  Triangulation tri;
  Matrix U, W;
  Vector W_norm;

  // Results
  Matrix C, N;
};


template <class TFloat, unsigned int VDim>
class CurrentsAttachmentTerm
{
public:
  typedef PointSetHamiltonianSystem<TFloat, VDim> HSystem;
  typedef typename HSystem::Vector Vector;
  typedef typename HSystem::Matrix Matrix;

  enum Mode { CURRENTS, VARIFOLD };

  /** Triangulation data type */
  typedef vnl_matrix<int> Triangulation;

  /**
   * Intermediate and output data for calls to Currents norm and scalar product
   * computations
   */
  struct CurrentScalarProductData
  {
    // Partials with respect to centers and normals
    Matrix dE_dC, dE_dN;

    // Partials with respect to the weights/areas (varifold only)
    Vector dE_dW;

    // Per-vertex energy component
    Vector z;

    // Indices of matrix indices in the upper diagonal
    std::vector<int> ut_i, ut_j;
  };


  /**
   * Constructor
   *   mode         : Mode (currents or varifolds)
   *   m            : Number of total landmarks (control and template)
   *   qT           : Target vertices
   *   tri_template : Triangles (3D) or edges (2D) of the template
   *   tri_target   : Triangles (3D) or edges (2D) of the target
   *   lab_template : Template label posterior array (per triangle)
   *   lab_target   : Target label posterior array (per triangle)
   */
  CurrentsAttachmentTerm(Mode mode, unsigned int m, const Matrix &qT,
                         const Triangulation &tri_template,
                         const Triangulation &tri_target,
                         const Matrix &lab_template, const Matrix &lab_target,
                         TFloat sigma, unsigned int n_threads)
    : tcan_template(tri_template, mode==VARIFOLD), tcan_target(tri_target, mode==VARIFOLD)
  {
    this->mode = mode;
    this->m = m;
    this->tri_target = tri_target;
    this->tri_template = tri_template;
    this->sigma = sigma;
    this->n_threads = n_threads;
    this->lab_template = lab_template;
    this->lab_target = lab_target;

    // Compute the triangle centers once and for all
    tcan_target.Forward(qT);

    // Setup the data structures for current gradient storage
    SetupCurrentScalarProductData(this->tri_target, this->cspd_target);
    SetupCurrentScalarProductData(this->tri_template, this->cspd_template);

    // Compute the current norm for the target (fixed quantity)
    cspd_target.z.fill(0.0);
    ComputeCurrentHalfNormSquared(tcan_target, cspd_target, lab_target, false);

    // Allocate this norm among template trianges equally (so that the z array
    // after the complete current computation makes sense on a per-triangle
    // basis and can be exported on a mesh surface)
    z0_template.set_size(tri_template.rows());
    z0_template.fill(cspd_target.z.sum() / tri_template.rows());
  }

  /**
   * Compute the 1/2 currents scalar product of a mesh with itself with optional
   * partial derivatives with respect to simplex centers and normals
   */
  void ComputeCurrentHalfNormSquared(
      TriangleCentersAndNormals<TFloat, VDim> &tcan,
      CurrentScalarProductData &cspd,
      const Matrix &label_matrix,
      bool grad)
  {
    // Compute the diagonal terms. These don't require any per-thread data because each
    // vertex can compute its own value and derivatives
    unsigned int nr = tcan.C.rows();
    TFloat f = -0.5 / (sigma * sigma);
    TFloat f_times_2 = 2.0 * f;
    int n_labels = label_matrix.columns();

    itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
    itk::ImageRegion<1> full_region({{0}}, {{nr}});
    mt->ParallelizeImageRegion<1>(
          full_region,
          [this,&tcan,&cspd,&label_matrix,n_labels,grad](const itk::ImageRegion<1> &thread_region)
      {
      // Get the list of rows to handle for this thread
      unsigned int r_begin = thread_region.GetIndex(0);
      unsigned int r_end = r_begin + thread_region.GetSize(0);

      // Get data arrays for fast memory access
      auto n_da = tcan.N.data_array();
      auto w_da = tcan.W_norm.data_block();
      auto dn_da = cspd.dE_dN.data_array();
      auto dw_da = cspd.dE_dW.data_block();
      auto z_da = cspd.z.data_block();
      auto l_da = label_matrix.data_array();

      // Handle the triangles for this thread
      for(unsigned int i = r_begin; i < r_end; i++)
        {
        const TFloat *ni = n_da[i], wi = w_da[i];
        TFloat *d_ni = dn_da[i], &d_wi = dw_da[i];
        const TFloat *l_i = l_da[i];

        TFloat label_weight = 0.0;
        for(int l = 0; l < n_labels; l++)
          label_weight += l_i[l]  * l_i[l];

        // Handle the diagonal term
        if(mode == CURRENTS)
          {
          for(unsigned int a = 0; a < VDim; a++)
            z_da[i] += 0.5 * label_weight * ni[a] * ni[a];

          // Handle the diagonal term in the gradient
          if(grad)
            for(unsigned int a = 0; a < VDim; a++)
              d_ni[a] += label_weight * ni[a];
          }
        else
          {
          // We know that the dot product of the normal with itself is just one
          // so we don't need to do anything with the normals. We just need to
          // compute the product of the weights
          z_da[i] += 0.5 * label_weight * wi * wi;
          if(grad)
            d_wi += label_weight * wi;
          }
        }
      }, nullptr);

    // Now thread over the upper triangle of the distance matrix. Threading is done over
    // all pairs of indices in the upper triangle. Thus to keep track of the objective
    // value z for each vertex and the gradients with respect to each vertex, we need
    // per-thread data that can be accumulated at the end
    itk::MultiThreaderBase::Pointer mt_ud = itk::MultiThreaderBase::New();
    std::mutex mutex_integration;
    itk::ImageRegion<1> full_region_ud({{0}}, {{cspd.ut_i.size()}});
    mt->ParallelizeImageRegion<1>(
          full_region_ud,
          [this,&tcan,&cspd,&label_matrix,n_labels,f_times_2,grad,nr,f, &mutex_integration](const itk::ImageRegion<1> &thread_region)
      {
      // Get the list of rows to handle for this thread
      unsigned int k_begin = thread_region.GetIndex(0);
      unsigned int k_end = k_begin + thread_region.GetSize(0);

      // Allocate the output data for this thread (wasteful)
      Matrix my_dE_dC(nr, 3, 0.0), my_dE_dN(nr, 3, 0.0);
      Vector my_dE_dW(nr, 0.0), my_z(nr, 0.0);

      // Get data arrays for fast memory access
      auto c_da = tcan.C.data_array();
      auto n_da = tcan.N.data_array();
      auto w_da = tcan.W_norm.data_block();
      auto dc_da = my_dE_dC.data_array();
      auto dn_da = my_dE_dN.data_array();
      auto dw_da = my_dE_dW.data_block();
      auto z_da = my_z.data_block();
      auto l_da = label_matrix.data_array();

      // Handle triangles for this thread
      for(unsigned int k = k_begin; k < k_end; k++)
        {
        // Get the indices for this pair
        int i = cspd.ut_i[k], j = cspd.ut_j[k];

        const TFloat *ci = c_da[i], *ni = n_da[i], wi = w_da[i];
        TFloat *d_ci = dc_da[i], *d_ni = dn_da[i], &d_wi = dw_da[i];
        const TFloat *l_i = l_da[i];
        const TFloat *cj = c_da[j], *nj = n_da[j], wj = w_da[j];
        TFloat *d_cj = dc_da[j], *d_nj = dn_da[j], &d_wj = dw_da[j];
        const TFloat *l_j = l_da[j];

        TFloat dq[VDim];
        TFloat dist_sq = 0.0;
        TFloat dot_ni_nj = 0.0;
        for(unsigned int a = 0; a < VDim; a++)
          {
          dq[a] = ci[a] - cj[a];
          dist_sq += dq[a] * dq[a];
          dot_ni_nj += ni[a] * nj[a];
          }

        TFloat label_weight = 0.0;
        for(int l = 0; l < n_labels; l++)
          label_weight += l_i[l] * l_j[l];

        TFloat K = label_weight * exp(f * dist_sq);

        if(mode == CURRENTS)
          {
          TFloat zij = K * dot_ni_nj;
          z_da[i] += zij;

          if(grad)
            {
            TFloat w = f_times_2 * zij;
            for(unsigned int a = 0; a < VDim; a++)
              {
              d_ci[a] += w * dq[a];
              d_cj[a] -= w * dq[a];
              d_ni[a] += K * nj[a];
              d_nj[a] += K * ni[a];
              }
            }
          }
        else
          {
          TFloat K_wi_wj = K * wi * wj;
          TFloat dot_ni_nj_sq = dot_ni_nj * dot_ni_nj;
          TFloat zij = K_wi_wj * dot_ni_nj_sq;
          z_da[i] += zij;

          if(grad)
            {
            TFloat w = f_times_2 * zij;
            for(unsigned int a = 0; a < VDim; a++)
              {
              d_ci[a] += w * dq[a];
              d_cj[a] -= w * dq[a];
              d_ni[a] += 2 * dot_ni_nj * K_wi_wj * nj[a];
              d_nj[a] += 2 * dot_ni_nj * K_wi_wj * ni[a];
              }
            d_wi += K * wj * dot_ni_nj_sq;
            d_wj += K * wi * dot_ni_nj_sq;
            }
          }
        }

      // Integrate using mutex
      std::lock_guard<std::mutex> guard(mutex_integration);
      cspd.dE_dC += my_dE_dC;
      cspd.dE_dN += my_dE_dN;
      cspd.dE_dW += my_dE_dW;
      cspd.z += my_z;
      }, nullptr);
  }

  /**
   * Compute the currents scalar product of two meshes with with optional
   * partial derivatives with respect to simplex centers and normals of the
   * first input
   */
  void ComputeCurrentScalarProduct(
      TriangleCentersAndNormals<TFloat, VDim> &tcan_1,
      TriangleCentersAndNormals<TFloat, VDim> &tcan_2,
      CurrentScalarProductData &cspd_1,
      const Matrix &label_matrix1, const Matrix &label_matrix2,
      bool grad)
  {
    int nr_1 = tcan_1.C.rows();

    // Multithread the vertices
    itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
    itk::ImageRegion<1> full_region({{0}}, {{static_cast<itk::SizeValueType>(nr_1)}});
    mt->ParallelizeImageRegion<1>(
          full_region,
          [this,&tcan_1,&tcan_2,&cspd_1,&label_matrix1,&label_matrix2,&grad](const itk::ImageRegion<1> &thread_region)
    {
      // Get data arrays for fast memory access
      auto c1_da = tcan_1.C.data_array(), c2_da = tcan_2.C.data_array();
      auto n1_da = tcan_1.N.data_array(), n2_da = tcan_2.N.data_array();
      auto w1_da = tcan_1.W_norm.data_block(), w2_da = tcan_2.W_norm.data_block();
      auto dc_da = cspd_1.dE_dC.data_array(), dn_da = cspd_1.dE_dN.data_array();
      auto dw_da = cspd_1.dE_dW.data_block();
      auto z1_da = cspd_1.z.data_block();
      auto l1_da = label_matrix1.data_array(), l2_da = label_matrix2.data_array();

      TFloat f = -0.5 / (sigma * sigma);
      TFloat f_times_2 = 2.0 * f;

      int n_labels = label_matrix1.columns();

      // Get the list of rows to handle for this thread
      unsigned int r_begin = thread_region.GetIndex(0);
      unsigned int r_end = r_begin + thread_region.GetSize(0);
      for(unsigned int i = r_begin; i < r_end; i++)
        {
        TFloat zi = 0.0;
        TFloat *ci = c1_da[i], *ni = n1_da[i], wi = w1_da[i];
        TFloat *d_ci = dc_da[i], *d_ni = dn_da[i], &d_wi = dw_da[i];
        TFloat dq[VDim];
        const TFloat *l_i = l1_da[i];

        for(unsigned int j = 0; j < tcan_2.C.rows(); j++)
          {
          // Compute the kernel
          TFloat *cj = c2_da[j], *nj = n2_da[j], wj = w2_da[j];
          const TFloat *l_j = l2_da[j];
          TFloat dist_sq = 0.0;
          TFloat dot_ni_nj = 0.0;
          for(unsigned int a = 0; a < VDim; a++)
            {
            dq[a] = ci[a] - cj[a];
            dist_sq += dq[a] * dq[a];
            dot_ni_nj += ni[a] * nj[a];
            }

          TFloat label_weight = 0.0;
          for(int l = 0; l < n_labels; l++)
            label_weight += l_i[l] * l_j[l];

          TFloat K = -label_weight * exp(f * dist_sq);

          if(mode == CURRENTS)
            {
            TFloat zij = K * dot_ni_nj;
            zi += zij;

            if(grad)
              {
              TFloat w = f_times_2 * zij;
              for(unsigned int a = 0; a < VDim; a++)
                {
                d_ci[a] += w * dq[a];
                d_ni[a] += K * nj[a];
                }
              }
            }
          else
            {
            TFloat K_wi_wj = K * wi * wj;
            TFloat dot_ni_nj_sq = dot_ni_nj * dot_ni_nj;
            TFloat zij = K_wi_wj * dot_ni_nj_sq;
            zi += zij;

            if(grad)
              {
              TFloat w = f_times_2 * zij;
              for(unsigned int a = 0; a < VDim; a++)
                {
                d_ci[a] += w * dq[a];
                d_ni[a] += 2 * dot_ni_nj * K_wi_wj * nj[a];
                }
              d_wi += K * wj * dot_ni_nj_sq;
              }
            }
          }

        // Store the cost for this template vertex
        z1_da[i] += zi;
        }
      }, nullptr);
  }

  /** Compute the energy and optional gradient of the energy */
  double Compute(const Matrix &q1, Matrix *grad = nullptr)
  {
    // Compute the triangle centers and normals
    tcan_template.Forward(q1);

    // Clear gradient terms if needed
    if(grad)
      {
      cspd_template.dE_dC.fill(0.0);
      cspd_template.dE_dN.fill(0.0);
      cspd_template.dE_dW.fill(0.0);
      }

    // Initialize the z-term
    cspd_template.z = z0_template;
    // double v0 = cspd_template.z.sum();

    // Add the squared norm term
    this->ComputeCurrentHalfNormSquared(tcan_template, cspd_template, lab_template, grad != nullptr);
    // double v1 = cspd_template.z.sum();

    // Subtract twice the scalar product term
    this->ComputeCurrentScalarProduct(tcan_template, tcan_target, cspd_template, lab_template, lab_target, grad != nullptr);
    // double v2 = cspd_template.z.sum();

    // Backpropagate the gradient to get gradient with respect to q1
    if(grad)
      tcan_template.Backward(cspd_template.dE_dC, cspd_template.dE_dN, cspd_template.dE_dW, *grad);

    // Return the total energy
    return cspd_template.z.sum();
  }

  /** Save the mesh representing current energy term */
  void SaveMesh(const Matrix &q1, const char *fname)
  {
    vtkSmartPointer<vtkPolyData> pd = vtkSmartPointer<vtkPolyData>::New();

    this->Compute(q1);

    // Assign points
    vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();
    for(unsigned int i = 0; i < m; i++)
      pts->InsertNextPoint(q1.data_array()[i]);
    pd->SetPoints(pts);

    // Assign polys
    vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
    for(unsigned int j = 0; j < tcan_template.tri.rows(); j++)
      {
      vtkIdType ids[VDim];
      for(unsigned int a = 0; a < VDim; a++)
        ids[a] = tcan_template.tri(j, a);
      polys->InsertNextCell(VDim, ids);
      }
    if(VDim == 3)
      pd->SetPolys(polys);
    else
      pd->SetLines(polys);

    // Create the energy array
    vtkSmartPointer<vtkDoubleArray> arr = vtkSmartPointer<vtkDoubleArray>::New();
    arr->SetName("CurrentEnergy");
    arr->SetNumberOfComponents(1);
    arr->SetNumberOfTuples(tcan_template.tri.rows());
    for(unsigned int j = 0; j < tcan_template.tri.rows(); j++)
      arr->SetComponent(j, 0, cspd_template.z[j]);
    pd->GetCellData()->AddArray(arr);

    WriteMesh(pd, fname);
  }

  /**
   * Initialize the data for scalar product computation (target or template)
   */
  void SetupCurrentScalarProductData(const Triangulation &tri,
                                     CurrentScalarProductData &cspd)
  {
    // Global objects
    int r = tri.rows();
    cspd.dE_dC.set_size(r, VDim);
    cspd.dE_dN.set_size(r, VDim);
    cspd.dE_dW.set_size(r);
    cspd.z.set_size(r);

    // Create a list of all the indices that require diagonal computation
    int nud = (r * r  - r) / 2;
    cspd.ut_i.reserve(nud);
    cspd.ut_j.reserve(nud);
    for(int i = 0; i < r; i++)
      {
      for(int j = i+1; j < r; j++)
        {
        cspd.ut_i.push_back(i);
        cspd.ut_j.push_back(j);
        }
      }
  }

protected:
  unsigned int m;
  Triangulation tri_template, tri_target;

  // Triangle quantity computer
  TriangleCentersAndNormals<TFloat, VDim> tcan_template, tcan_target;

  // Template and target current norm/scalar product data
  CurrentScalarProductData cspd_template, cspd_target;

  // Label posteriors
  Matrix lab_template, lab_target;

  // Current squared norm of the target allocated equally between all
  // the trianges in the template
  Vector z0_template;

  // Kernel sigma
  TFloat sigma;

  // Threading
  unsigned int n_threads;

  // Mode
  Mode mode;
};

template <class TFloat, unsigned int VDim>
class MeshJacobianPenaltyTerm
{
public:
  typedef PointSetHamiltonianSystem<TFloat, VDim> HSystem;
  typedef typename HSystem::Vector Vector;
  typedef typename HSystem::Matrix Matrix;
  typedef vnl_matrix<int> Triangulation;

  /**
   * Constructor
   *   qR           : Reference vertices (Jacobian computed relative to them)
   *   tri          : Triangles (3D) or edges (2D) of the mesh
   */
  MeshJacobianPenaltyTerm(const Matrix &qR, const Triangulation &tri)
    : tcan_ref(tri, true), tcan_def(tri, true)
  {
    tcan_ref.Forward(qR);

    dE_dC.set_size(tcan_ref.W_norm.size(), VDim); dE_dC.fill(0.0);
    dE_dN.set_size(tcan_ref.W_norm.size(), VDim); dE_dN.fill(0.0);
    dE_dW.set_size(tcan_ref.W_norm.size()); dE_dW.fill(0.0);
  }

  /**
   * Compute the Jacobian term
   */
  double Compute(const Matrix &q1, Matrix *grad = nullptr)
  {
    // Compute the triangle centers and normals for the template.
    // TODO: this is redundant because same is being done by the varifold term
    tcan_def.Forward(q1);

    // Compute the change in triangle areas
    double penalty = 0.0;
    double z = 2.0 / std::log(10);
    for(unsigned int i = 0; i < tcan_def.W_norm.size(); i++)
      {
      double area_def = tcan_def.W_norm[i];
      double area_ref = tcan_ref.W_norm[i];
      double log_jac = std::log10(area_def / area_ref);
      penalty += log_jac * log_jac;

      if(grad)
        dE_dW[i] = z * log_jac / area_def;
      }

    if(grad)
      tcan_def.Backward(dE_dC, dE_dN, dE_dW, *grad);

    return penalty;
  }

protected:
  // Triangle quantity computer for the reference and deforming meshes
  TriangleCentersAndNormals<TFloat, VDim> tcan_ref, tcan_def;

  // Partials with respect to centers and normals
  Matrix dE_dC, dE_dN;

  // Partials with respect to the weights/areas (varifold only)
  Vector dE_dW;
};

/*
template <class TFloat, unsigned int VDim>
struct QuaternionTraits
{
  using Vec = vnl_vector_fixed<TFloat, VDim>;
  static Vec cross(const Vec &v1, const Vec &v2) { throw std::exception(); }
  static Vec dot(const Vec &v1, const Vec &v2) { throw std::exception(); }
  static int read(const double *array, Vec &v) { throw std::exception();  }
  static int write(const Vec &v, double *array) { throw std::exception(); }
  static constexpr int size = VDim;
};

template <class TFloat>
struct QuaternionTraits<TFloat, 3>
{
  using Vec = vnl_vector_fixed<TFloat, 3>;
  static Vec cross(const Vec &v1, const Vec &v2) { return vnl_cross_3d(v1, v2); }
  static Vec dot(const Vec &v1, const Vec &v2) { return dot_product(v1, v2); }
  static int read(const double *array, Vec &v)
    { v[0] = array[0]; v[1] = array[1]; v[2] = array[2];  return 3; }
  static int write(const Vec &v, double *array)
    { array[0] = v[0]; array[1] = v[1]; array[2] = v[2];  return 3; }
  static constexpr int size = 3;
};

template <class TFloat>
struct QuaternionTraits<TFloat, 2>
{
  using Vec = TFloat;
  static Vec cross(const Vec &v1, const Vec &v2) { return v1 * v2; }
  static Vec dot(const Vec &v1, const Vec &v2) { return 0; }
  static int read(const double *array, Vec &v) { v = array[0]; return 1; }
  static int write(const Vec &v, double *array) { array[0] = v; return 1; }
  static constexpr int size = 1;
};
*/


template <class TFloat>
struct Quaternion
{
  using Self = Quaternion<TFloat>;
  using Vec = vnl_vector_fixed<TFloat, 3>;

  Quaternion(TFloat r, const Vec &v) : r(r), v(v) {}

  Quaternion() : r(0.0), v(0.0) {}

  /*
  int read_from_array(const double *array)
    {
    r = array[0];
    return 1 + Traits::read(array, v);
    }

  int write_to_array(double *array)
    {
    array[0] = r;
    return 1 + Traits::write(v, array);
    }

  static constexpr int size = 1 + Traits::size;
  */

  static Self mult_q1_q2(const Self &q1, const Self &q2)
    {
    return Self(q1.r * q2.r - dot_product(q1.v, q2.v),
                q1.v * q2.r + q2.v * q1.r + vnl_cross_3d(q1.v, q2.v));
    }

  static Self mult_q1_q2conj(const Self &q1, const Self &q2)
    {
    return Self(q1.r * q2.r + dot_product(q1.v, q2.v),
                q1.v * q2.r - q2.v * q1.r - vnl_cross_3d(q1.v, q2.v));
    }

  static Self mult_q1conj_q2(const Self &q1, const Self &q2)
    {
    return Self(q1.r * q2.r + dot_product(q1.v, q2.v),
                - q1.v * q2.r + q2.v * q1.r - vnl_cross_3d(q1.v, q2.v));
    }

  static Self mult_q1_x_q2conj(const Self &q1, const Self &x, const Self &q2)
    {
    return mult_q1_q2conj(mult_q1_q2(q1, x), q2);
    }

  static Self mult_q1conj_x_q2(const Self &q1, const Self &x, const Self &q2)
    {
    return mult_q1_q2(mult_q1conj_q2(q1, x), q2);
    }

  static Self scale(const Self &q, TFloat a)
    {
    return Self(q.r * a, q.v * a);
    }

  TFloat r;
  Vec v;
};

template <class TFloat, unsigned int VDim>
struct QuaternionRotationTraits
{

};

template <class TFloat>
struct QuaternionRotationTraits<TFloat, 3>
{
  typedef vnl_vector_fixed<TFloat, 3> Vec;
  typedef Quaternion<TFloat> Q;

  static Q point_to_quaternion(const Vec &x) { return Q(0, x); }
  static Vec quaternion_to_point(const Q &q) { return q.v; }
  static Q zero_rotation()
    {
    return Q(1.0, typename Q::Vec(0., 0., 0.));
    }
  static Q coeff_to_quaternion(const double  *arr)
    {
    return Q(arr[0], typename Q::Vec(arr[1], arr[2], arr[3]));
    }
  static void quaternion_to_coeff(const Q&q, double *arr)
    {
    arr[0] = q.r; arr[1] = q.v[0]; arr[2] = q.v[1]; arr[3] = q.v[2];
    }

  static constexpr int size = 4;
};

template <class TFloat>
struct QuaternionRotationTraits<TFloat, 2>
{
  typedef vnl_vector_fixed<TFloat, 2> Vec;
  typedef Quaternion<TFloat> Q;

  static Q point_to_quaternion(const Vec &x) { return Q(0, typename Q::Vec(x[0], x[1], 0)); }
  static Vec quaternion_to_point(const Q &q) { return Vec(q.v[0], q.v[1]); }
  static Q zero_rotation()
    {
    return Q(1.0, typename Q::Vec(0., 0., 0.));
    }
  static Q coeff_to_quaternion(const double *arr)
    {
    return Q(arr[0], typename Q::Vec(0, 0, arr[1]));
    }
  static void quaternion_to_coeff(const Q&q, double *arr)
    {
    arr[0] = q.r; arr[1] = q.v[2];
    }

  static constexpr int size = 2;
};



template <class TFloat, unsigned int VDim>
class QuaternionTransform
{
public:
  typedef vnl_vector_fixed<TFloat, VDim> Vec;
  typedef Quaternion<TFloat> Q;
  typedef QuaternionRotationTraits<TFloat, VDim> QRTraits;
  typedef vnl_matrix<TFloat> Matrix;
  typedef QuaternionTransform<TFloat, VDim> Self;

  QuaternionTransform(const Matrix &X)
  {
    // Store the fixed coordinates
    this->X = X;
    n = X.rows();

    // Compute the center and the extents of the coordinates for centering/scaling
    Vec extent;
    for(unsigned int a = 0; a < VDim; a++)
      {
      auto col = X.get_column(a);
      center[a] = col.mean();
      extent[a] = col.max_value() - col.min_value();
      }
    diameter = extent.max_value();
  }

  void CopyCenterAndDiameter(const Self &other)
    {
    this->center = other.center;
    this->diameter = other.diameter;
    }

  // Apply transform to a set of coordiantes X
  void Forward(const Q &q, const Vec &b, Matrix &Y)
    {
      // Apply quaternion to each point
      auto v = q.v;
      auto r = q.r;
      for(unsigned int i = 0; i < n; i++)
        {
        Q x_i = QRTraits::point_to_quaternion(X.get_row(i) - center);
        auto p = x_i.v;
        // Q z_i(0, v * dot_product(v, p) + r * r * p + 2 * r * vnl_cross_3d(v, p) - vnl_cross_3d(vnl_cross_3d(v, p), v));

        Q z_i(0, v * dot_product(v, p) + r * r * p + 2 * r * vnl_cross_3d(v, p) - vnl_cross_3d(vnl_cross_3d(v, p), v));

        // Q z_i_2 = Q::mult_q1_q2conj(Q::mult_q1_q2(q, x_i), q);
        Vec y_i = QRTraits::quaternion_to_point(z_i) + center + b * diameter;
        Y.set_row(i, y_i.as_ref());
        }
    }

  // Backpropagate the gradient of some loss F with respect to Y onto the quaternion coefficients
  void Backward(const Q &q, const Vec &b,
                       const Matrix &df_dY, Q &df_dq, Vec &df_db)
    {
    // Initialize return values
    df_dq = Q();
    df_db.fill(0.0);

    // Apply quaternion to each point
    auto v = q.v;
    auto r = q.r;
    for(unsigned int i = 0; i < n; i++)
      {
      Vec df_dY_i = df_dY.get_row(i);
      df_db += df_dY_i * diameter;

      Q x_i = QRTraits::point_to_quaternion(X.get_row(i) - center);
      Q q_df_dY_i = QRTraits::point_to_quaternion(df_dY_i);
      auto p = x_i.v, dp = q_df_dY_i.v;

      df_dq.r += dot_product(dp, 2 * r * p) + 2 * dot_product(dp, vnl_cross_3d(v, p));
      df_dq.v += dp * dot_product(v, p) + p * dot_product(dp, v);
      df_dq.v += 2 * r * vnl_cross_3d(p, dp);
      df_dq.v -= vnl_cross_3d(vnl_cross_3d(dp, v), p) + vnl_cross_3d(vnl_cross_3d(p, v), dp);
      }
    }

private:
  unsigned int n;
  Matrix X;
  Vec center;
  TFloat diameter;
};

/**
 * @brief Bidirectional similarity transform of two point sets based on quaternions.
 * See quat.ipynb for the computation of these transformations and their derivatives.
 */
template <class TFloat, unsigned int VDim>
class BidirectionalQuaternionTransform
{
public:
  typedef vnl_vector_fixed<TFloat, VDim> Vec;
  typedef Quaternion<TFloat> Q;
  typedef QuaternionRotationTraits<TFloat, VDim> QRTraits;
  typedef vnl_matrix<TFloat> Matrix;
  typedef BidirectionalQuaternionTransform<TFloat, VDim> Self;

  BidirectionalQuaternionTransform(const Matrix &X0, const Matrix &XT)
  {
    // Store the fixed coordinates
    this->X0 = X0;
    this->XT = XT;
    n0 = X0.rows();
    nT = XT.rows();

    // Compute the centers and the extents of the template's coordinates
    Vec extent;
    for(unsigned int a = 0; a < VDim; a++)
      {
      // Template
      auto col = X0.get_column(a);
      C0[a] = col.mean();
      extent[a] = col.max_value() - col.min_value();

      // Target
      CT[a] = XT.get_column(a).mean();
      }
    diameter = extent.max_value();
  }

  // Get translation that matches centers
  Vec GetCenterMatrchTranslation()
    {
    return (CT - C0) / diameter;
    }

  // Transform a single point from template to target
  Vec TransformPoint(const Q &q, const Vec &b, const Vec &X)
  {
    Q qx_i = QRTraits::point_to_quaternion(X - C0);
    Q qy_i = Q::mult_q1_x_q2conj(q, qx_i, q);
    Vec y_i = QRTraits::quaternion_to_point(qy_i) + C0 + b * diameter;
    return y_i;
  }

  // Apply transform to a set of coordiantes X
  void Forward(const Q &q, const Vec &b, Matrix &Y0, Matrix &YT)
    {
      // Transform the template towards the target
      for(unsigned int i = 0; i < n0; i++)
        {
        Vec y_i = TransformPoint(q, b, X0.get_row(i));
        Y0.set_row(i, y_i.as_ref());
        }

      // Get the squared norm of the quaternion
      TFloat q_norm_sq = Q::mult_q1_q2conj(q, q).r;

      // Transform the target towards the template
      for(unsigned int i = 0; i < nT; i++)
        {
        Q qx_i = QRTraits::point_to_quaternion(XT.get_row(i) - C0 - b * diameter);
        Q qy_i = Q::scale(Q::mult_q1conj_x_q2(q, qx_i, q), 1.0 / (q_norm_sq * q_norm_sq));
        Vec y_i = QRTraits::quaternion_to_point(qy_i) + C0;
        YT.set_row(i, y_i.as_ref());
        }
    }

  // Backpropagate the gradient of some loss F with respect to Y onto the quaternion coefficients
  void Backward(const Q &q, const Vec &b,
                const Matrix &df_dY0, const Matrix &df_dYT,
                Q &df_dq, Vec &df_db)
    {
    // Initialize return values
    df_dq = Q();
    df_db.fill(0.0);

    // Backprop the forward transform
    for(unsigned int i = 0; i < n0; i++)
      {
      Vec gamma = df_dY0.get_row(i);
      df_db += gamma * diameter;

      Q x_i = QRTraits::point_to_quaternion(X0.get_row(i) - C0);
      Q q_gamma = QRTraits::point_to_quaternion(gamma);
      Q grad_q = Q::scale(Q::mult_q1_x_q2conj(q_gamma, q, x_i), 2.0);
      df_dq.r += grad_q.r; df_dq.v += grad_q.v;
      }

    // Get the squared norm of the quaternion
    TFloat q_norm_sq = Q::mult_q1_q2conj(q, q).r;
    TFloat q_norm_4 = q_norm_sq * q_norm_sq;
    TFloat q_norm_6 = q_norm_4 * q_norm_sq;

    // Backprop the backward transform
    for(unsigned int i = 0; i < nT; i++)
      {
      Vec gamma = df_dYT.get_row(i);
      Q q_gamma = QRTraits::point_to_quaternion(gamma);
      Q x_i = QRTraits::point_to_quaternion(XT.get_row(i) - C0 - b * diameter);

      df_db -= QRTraits::quaternion_to_point(Q::mult_q1_x_q2conj(q, q_gamma, q)) * (diameter / q_norm_4);

      Q grad_q_t1 = Q::scale(Q::mult_q1_x_q2conj(x_i, q, q_gamma), 2.0 / q_norm_4);
      Vec z = QRTraits::quaternion_to_point(Q::mult_q1conj_x_q2(q, x_i, q));
      Q grad_q_t2 = Q::scale(q, -4.0 * dot_product(z, gamma) / q_norm_6);
      df_dq.r += grad_q_t1.r + grad_q_t2.r;
      df_dq.v += grad_q_t1.v + grad_q_t2.v;
      }
    }

private:
  unsigned int n0, nT;
  Matrix X0, XT;
  Vec C0, CT;
  TFloat diameter;
};


template <class TFloat, unsigned int VDim>
class PointSetSimilarityMatchingCostFunction : public vnl_cost_function
{
public:
  typedef vnl_matrix<int> Triangulation;

  typedef QuaternionTransform<TFloat, VDim> QT;
  typedef typename QT::QRTraits QRTraits;
  typedef typename QT::Q Quaternion;
  typedef typename QT::Vec Vec;
  typedef typename QT::Matrix Matrix;

  // Separate type because vnl optimizer is double-only
  typedef std::tuple<Quaternion, Vec> CoeffType;

  PointSetSimilarityMatchingCostFunction(
    const ShootingParameters &param,
      const Matrix &q0, const Matrix &qT,
      Triangulation tri_template,
      Triangulation tri_target,
      const Matrix &lab_template, const Matrix &lab_target)
    : vnl_cost_function(QRTraits::size + VDim), qtran(q0, qT)
    {
    this->q0 = q0;
    this->qT = qT;
    this->param = param;
    this->k = p0.rows();
    this->m0 = q0.rows();
    this->mT = qT.rows();
    this->q1.set_size(m0, VDim);
    this->qT_1.set_size(mT, VDim);
    this->gamma0.set_size(m0, VDim);
    this->gammaT.set_size(mT, VDim);

    // Set up the currents attachment terms in both directions
    ca_temp_to_targ = nullptr;
    ca_targ_to_temp = nullptr;
    if(param.attach == ShootingParameters::Current || param.attach == ShootingParameters::Varifold)
      {
      // Create the appropriate attachment terms (currents or varifolds)
      ca_temp_to_targ = new CATerm(
            param.attach == ShootingParameters::Current ? CATerm::CURRENTS : CATerm::VARIFOLD,
            m0, qT, tri_template, tri_target, lab_template, lab_target,
            param.currents_sigma, param.n_threads);

      ca_targ_to_temp = new CATerm(
            param.attach == ShootingParameters::Current ? CATerm::CURRENTS : CATerm::VARIFOLD,
            mT, q0, tri_target, tri_template, lab_target, lab_template,
            param.currents_sigma, param.n_threads);
      }
    }

  ~PointSetSimilarityMatchingCostFunction()
  {
    if(ca_temp_to_targ)
      delete ca_temp_to_targ;
    if(ca_targ_to_temp)
      delete ca_targ_to_temp;
  }

  CoeffType ArrayToCoeff(const double *arr)
    {
    Quaternion q = QRTraits::coeff_to_quaternion(arr);
    Vec b;
    for(unsigned int a = 0; a < VDim; a++)
      b[a] = arr[a + QRTraits::size];
    return std::make_tuple(q, b);
    }

  void CoeffToArray(const CoeffType &c, double *arr)
    {
    QRTraits::quaternion_to_coeff(std::get<0>(c), arr);
    for(unsigned int a = 0; a < VDim; a++)
      arr[a + QRTraits::size] = std::get<1>(c)[a];
    }

  CoeffType InitialSolution()
    {
    Quaternion q = QRTraits::zero_rotation();
    Vec b = qtran.GetCenterMatrchTranslation();
    return std::make_tuple(q, b);
    }

  virtual double ComputeEuclideanAttachment()
  {
    // Compute the landmark errors
    double E_data = 0.0;
    unsigned int i_temp = (k == m0) ? 0 : k;
    gamma0.fill(0.0);
    for(unsigned int a = 0; a < VDim; a++)
      {
      for(unsigned int i = i_temp; i < m0; i++)
        {
        gamma0(i,a) = q1(i,a) - qT(i - i_temp, a);
        E_data += 0.5 * gamma0(i,a) * gamma0(i,a);
        }
      }

    return E_data;
  }

  virtual void ComputeTransformedPoints(const CoeffType &c, Matrix &q1)
    {
    // From x, extract quaterion
    Quaternion w; Vec b;
    std::tie(w, b) = c;
    q1.set_size(q0.rows(), q0.columns());

    Matrix qT_1(qT.rows(), qT.columns());
    qtran.Forward(w, b, q1, qT_1);
    }

  virtual Matrix ComputeAffineMatrix(const CoeffType &c)
  {
    // From x, extract quaterion
    Quaternion w; Vec b;
    std::tie(w, b) = c;

    // Initialize the affine matrix
    Matrix A(VDim+1, VDim+1);
    A.set_identity();

    // Apply the transform to the first VDim columns, this will give us b in the last row,
    // and Rx+b in the other rows.
    Matrix X = A.extract(VDim+1, VDim);
    auto b_aff = qtran.TransformPoint(w, b, X.get_row(VDim));
    for(unsigned int i = 0; i < VDim; i++)
      {
      auto y = qtran.TransformPoint(w, b, X.get_row(i));
      for(unsigned int j = 0; j < VDim; j++)
        A(j,i) =  y[j] - b_aff[j];
      A(i,VDim) = b_aff[i];
      }

    return A;
  }

  virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g)
    {
    // From x, extract quaterion
    Quaternion w; Vec b;
    std::tie(w, b) = ArrayToCoeff(x.data_block());

    // Transform the coordinates from q0 to q1 and from qT to qT_1
    qtran.Forward(w, b, q1, qT_1);

    // Compute the data attachment term
    double E_temp_to_targ = 0.0, E_targ_to_temp = 0.0;
    if(param.attach == ShootingParameters::Euclidean)
      {
      E_temp_to_targ = ComputeEuclideanAttachment();
      }
    else if(param.attach == ShootingParameters::Current || param.attach == ShootingParameters::Varifold)
      {
      if(g)
        {
        E_temp_to_targ = this->ca_temp_to_targ->Compute(q1, &gamma0);
        E_targ_to_temp = this->ca_targ_to_temp->Compute(qT_1, &gammaT);
        }
      else
        {
        E_temp_to_targ = this->ca_temp_to_targ->Compute(q1);
        E_targ_to_temp = this->ca_targ_to_temp->Compute(qT_1);
        }
      }

    if(f)
      *f = E_temp_to_targ + E_targ_to_temp;

    if(g)
      {
      // Flow alpha back
      Quaternion df_dw; Vec df_db;
      qtran.Backward(w, b, gamma0, gammaT, df_dw, df_db);

      // Pack the gradient into the output vector
      CoeffToArray(std::make_tuple(df_dw, df_db), g->data_block());

      // Count this as an iteration
      ++iter;
      }

    // Print current results
    if(verbose && g && f)
      {
      printf("It = %04d  tmp_2_trg = %8.2f  trg_2_tmp = %8.2f  total = %8.2f\n", iter, E_temp_to_targ, E_targ_to_temp, *f);
      }
    }

 void SetVerbose(bool value)
   {
   this->verbose = value;
   }

protected:
  ShootingParameters param;
  Matrix qT, p0, q0, p1, q1, qT_1;
  Matrix gamma0, gammaT;

  // Quaternion math
  BidirectionalQuaternionTransform<TFloat, VDim> qtran;

  // Attachment terms
  typedef CurrentsAttachmentTerm<TFloat, VDim> CATerm;
  CATerm *ca_targ_to_temp, *ca_temp_to_targ;

  // Number of control points (k) and total points (m)
  unsigned int k, m0, mT;

  // Whether to print values at each iteration
  bool verbose = false;
  unsigned int iter = 0;
};







template <class TFloat, unsigned int VDim>
class PointSetShootingCostFunction : public vnl_cost_function
{
public:
  typedef PointSetHamiltonianSystem<TFloat, VDim> HSystem;
  typedef typename HSystem::Vector Vector;
  typedef typename HSystem::Matrix Matrix;
  typedef vnl_matrix<int> Triangulation;

  // Separate type because vnl optimizer is double-only
  typedef vnl_vector<double> DVector;

  PointSetShootingCostFunction(
    const ShootingParameters &param,
      const Matrix &q0, const Matrix &p0, const Matrix &qT,
      Triangulation tri_template,
      Triangulation tri_target,
      const Matrix &lab_template, const Matrix &lab_target)
    : vnl_cost_function(p0.rows() * VDim),
      hsys(q0, param.sigma, param.N, q0.rows() - p0.rows(), param.n_threads)
    {
    this->p0 = p0;
    this->q0 = q0;
    this->qT = qT;
    this->param = param;
    this->k = p0.rows();
    this->m = q0.rows();
    this->p1.set_size(k,VDim);
    this->q1.set_size(m,VDim);

    for(unsigned int a = 0; a < VDim; a++)
      {
      alpha[a].set_size(m);
      beta[a].set_size(k); beta[a].fill(0.0);
      grad_f[a].set_size(m);
      }

    // Set up Ralston integration
    this->hsys.SetRalstonIntegration(param.use_ralston_method);

    // Set up the currents attachment
    currents_attachment = nullptr;
    if(param.attach == ShootingParameters::Current || param.attach == ShootingParameters::Varifold)
      {
      // Create the appropriate attachment term (currents or varifolds)
      currents_attachment = new CATerm(
            param.attach == ShootingParameters::Current ? CATerm::CURRENTS : CATerm::VARIFOLD,
            m, qT, tri_template, tri_target, lab_template, lab_target,
            param.currents_sigma, param.n_threads);

      // Allocate the gradient storage
      grad_currents.set_size(m, VDim);
      }

    // Set up the jacobian term
    if(param.w_jacobian > 0)
      {
      this->jacobian_term = new JacobianTerm(q0, tri_template);
      this->grad_jacobian.set_size(m, VDim);
      }
    }

  ~PointSetShootingCostFunction()
  {
    if(currents_attachment)
      delete currents_attachment;
  }

  DVector wide_to_tall(const Vector p[VDim])
    {
    DVector v(p[0].size() * VDim);
    int pos = 0;
    for(unsigned int a = 0; a < VDim; a++)
      for(unsigned int i = 0; i < k; i++)
        v[pos++] = p[a](i);
    return v;
    }

  DVector wide_to_tall(const Matrix &p)
    {
    DVector v(p.rows() * VDim);
    int pos = 0;
    for(unsigned int a = 0; a < VDim; a++)
      for(unsigned int i = 0; i < k; i++)
        v[pos++] = p(i,a);
    return v;
    }

  Matrix tall_to_wide(const DVector &v)
    {
    Matrix p(v.size() / VDim, VDim);
    int pos = 0;
    for(unsigned int a = 0; a < VDim; a++)
      for(unsigned int i = 0; i < k; i++)
        p(i,a) = (TFloat) v[pos++];
    return p;
    }

  virtual double ComputeEuclideanAttachment()
  {
    // Compute the landmark errors
    double E_data = 0.0;
    unsigned int i_temp = (k == m) ? 0 : k;
    for(unsigned int a = 0; a < VDim; a++)
      {
      alpha[a].fill(0.0);
      for(unsigned int i = i_temp; i < m; i++)
        {
        alpha[a](i) = q1(i,a) - qT(i - i_temp, a);
        E_data += 0.5 * alpha[a](i) * alpha[a](i);
        }
      }

    return E_data;
  }

 virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g)
    {
    // Initialize the p0-vector
    p0 = tall_to_wide(x);

    // Perform flow
    double H = hsys.FlowHamiltonian(p0, q1, p1);

    // Compute the data attachment term
    double E_data = 0.0;
    if(param.attach == ShootingParameters::Euclidean)
      {
      E_data = ComputeEuclideanAttachment();
      for(unsigned int i = 0; i < m; i++)
        for(unsigned int a = 0; a < VDim; a++)
          alpha[a][i] *= param.lambda;
      }
    else if(param.attach == ShootingParameters::Current || param.attach == ShootingParameters::Varifold)
      {
      static int my_iter = 0;
      if(g)
        {
        E_data = currents_attachment->Compute(q1, &grad_currents);
        for(unsigned int i = 0; i < m; i++)
          for(unsigned int a = 0; a < VDim; a++)
            alpha[a][i] = param.lambda * grad_currents(i,a);
        }
      else
        E_data = currents_attachment->Compute(q1);
      }

    // Compute the mesh Jacobian (regularization) term
    double E_jac = 0.0;
    if(param.w_jacobian > 0)
      {
      E_jac = jacobian_term->Compute(q1, &grad_jacobian);
      for(unsigned int i = 0; i < m; i++)
        for(unsigned int a = 0; a < VDim; a++)
          alpha[a][i] += param.w_jacobian * grad_jacobian(i,a);
      }

    if(f)
      *f = param.gamma * H + param.lambda * E_data + param.w_jacobian * E_jac;

    if(g)
      {
      // Multiply gradient of f. wrt q1 (alpha) by Jacobian of q1 wrt p0
      hsys.FlowGradientBackward(alpha, beta, grad_f);

      // Recompute Hq/Hp at initial timepoint (TODO: why are we doing this?)
      hsys.ComputeHamiltonianJet(q0, p0, false);

      // Complete gradient computation
      for(unsigned int a = 0; a < VDim; a++)
        {
        // Combine the gradient terms
        grad_f[a] += hsys.GetHp(a).extract(k) * param.gamma;
        }

      // Pack the gradient into the output vector
      *g = wide_to_tall(grad_f);

      // Count this as an iteration
      ++iter;
      }

    // Print current results
    if(verbose && g && f) 
      {
      printf("It = %04d  H = %8.2f  DA = %8.2f  JC = %8.2f  f = %8.2f\n",
             iter, H * param.gamma, E_data * param.lambda, E_jac * param.w_jacobian, *f);
      }
    }

 void SetVerbose(bool value) 
   { 
   this->verbose = value;
   }

protected:
  HSystem hsys;
  ShootingParameters param;
  Matrix qT, p0, q0, p1, q1;
  Vector alpha[VDim], beta[VDim], grad_f[VDim];

  // Attachment terms
  typedef CurrentsAttachmentTerm<TFloat, VDim> CATerm;
  CATerm *currents_attachment;

  // Penalty terms
  typedef MeshJacobianPenaltyTerm<TFloat, VDim> JacobianTerm;
  JacobianTerm *jacobian_term;

  // For currents, the gradient is supplied as a matrix
  Matrix grad_currents;

  // For the Jacobian term, the gradient is also stored in a matrix
  Matrix grad_jacobian;

  // Number of control points (k) and total points (m)
  unsigned int k, m;

  // Whether to print values at each iteration
  bool verbose = false;
  unsigned int iter = 0;
};



template <class TFloat, unsigned int VDim>
class PointSetShootingLineSearchCostFunction : public vnl_cost_function
{
public:
  typedef PointSetHamiltonianSystem<TFloat, VDim> HSystem;
  typedef typename HSystem::Vector Vector;
  typedef typename HSystem::Matrix Matrix;

  // Separate type because vnl optimizer is double-only
  typedef vnl_vector<double> DVector;

  PointSetShootingLineSearchCostFunction(
    const ShootingParameters &param, const Matrix &q0, const Matrix &p0, const Matrix &qT, const Matrix &del_p0)
    : vnl_cost_function(q0.rows() * VDim),
      hsys(q0, param.sigma, param.N, 0, param.n_threads)
    {
    this->p0 = p0;
    this->del_p0 = del_p0;
    this->qT = qT;
    this->param = param;
    this->k = q0.rows();
    this->p1.set_size(k,VDim);
    this->q1.set_size(k,VDim);
    hsys.SetRalstonIntegration(param.use_ralston_method);
    }

  virtual double f (vnl_vector<double> const& x)
    {
    TFloat alpha = (TFloat) x[0];

    // Perform flow
    double H = hsys.FlowHamiltonian(p0 + alpha * del_p0, q1, p1);

    // Compute the landmark errors
    double fnorm_sq = 0.0;
    for(unsigned int a = 0; a < VDim; a++)
      {
      for(unsigned int i = 0; i < k; i++)
        {
        double d = q1(i,a) - qT(i,a);
        fnorm_sq += d * d;
        }
      }

    // Compute the landmark part of the objective
    double Edist = 0.5 * fnorm_sq;

    // cout << H + param.lambda * Edist << endl;
    return H + param.lambda * Edist;
    }



protected:
  HSystem hsys;
  ShootingParameters param;
  Matrix qT, p0, del_p0, q0, p1, q1;
  unsigned int k;
};



template <class TFloat, unsigned int VDim>
class PointSetShootingTransversalityCostFunction : public vnl_cost_function
{
public:
  typedef PointSetHamiltonianSystem<TFloat, VDim> HSystem;
  typedef typename HSystem::Vector Vector;
  typedef typename HSystem::Matrix Matrix;

  // Separate type because vnl optimizer is double-only
  typedef vnl_vector<double> DVector;

  PointSetShootingTransversalityCostFunction(
    const ShootingParameters &param, const Matrix &q0, const Matrix &qT)
    : vnl_cost_function(q0.rows() * VDim),
      hsys(q0, param.sigma, param.N, 0, param.n_threads)
    {
    this->p0 = (qT - q0) / param.N;
    this->qT = qT;
    this->param = param;
    this->k = q0.rows();
    this->p1.set_size(k,VDim);
    this->q1.set_size(k,VDim);
    hsys.SetRalstonIntegration(param.use_ralston_method);

    for(unsigned int a = 0; a < VDim; a++)
      {
      alpha[a].set_size(k);
      beta[a].set_size(k); beta[a].fill(0.0);
      G[a].set_size(k);
      grad_f[a].set_size(k);
      }
    }

  DVector wide_to_tall(const Vector p[VDim])
    {
    DVector v(k * VDim);
    int pos = 0;
    for(unsigned int a = 0; a < VDim; a++)
      for(unsigned int i = 0; i < k; i++)
        v[pos++] = p[a](i);
    return v;
    }

  DVector wide_to_tall(const Matrix &p)
    {
    DVector v(k * VDim);
    int pos = 0;
    for(unsigned int a = 0; a < VDim; a++)
      for(unsigned int i = 0; i < k; i++)
        v[pos++] = p(i,a);
    return v;
    }

  Matrix tall_to_wide(const DVector &v)
    {
    Matrix p(k,VDim);
    int pos = 0;
    for(unsigned int a = 0; a < VDim; a++)
      for(unsigned int i = 0; i < k; i++)
        p(i,a) = (TFloat) v[pos++];
    return p;
    }

  virtual void compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g)
    {
    // Initialize the p0-vector
    p0 = tall_to_wide(x);

    // Perform flow
    double H = hsys.FlowHamiltonian(p0, q1, p1);

    // Compute G and alpha/beta
    double Gnorm_sq = 0.0, dsq = 0.0;
    for(unsigned int a = 0; a < VDim; a++)
      {
      for(unsigned int i = 0; i < k; i++)
        {
        G[a](i) = p1(i, a) + param.lambda * (q1(i, a) - qT(i, a));
        Gnorm_sq += G[a][i] * G[a][i];
        dsq += (q1(i, a) - qT(i, a)) * (q1(i, a) - qT(i, a));

        alpha[a](i) = param.lambda * G[a][i];
        beta[a](i) = G[a][i];
        }
      }

    if(f)
      *f = 0.5 * Gnorm_sq;

    if(g)
      {
      // Multiply gradient of f. wrt q1 (alpha) by Jacobian of q1 wrt p0
      hsys.FlowGradientBackward(alpha, beta, grad_f);

      // Pack the gradient into the output vector
      *g = wide_to_tall(grad_f);
      }

    // Print the current state
    printf("H=%8.6f   Edist=%8.6f   E=%8.6f   |G|=%8.6f\n",
      H, 0.5 * param.lambda * dsq, H + 0.5 * param.lambda * dsq, sqrt(Gnorm_sq));

    }



protected:
  HSystem hsys;
  ShootingParameters param;
  Matrix qT, p0, q0, p1, q1;
  Vector alpha[VDim], beta[VDim], G[VDim], grad_f[VDim];
  unsigned int k;

};



#include <vnl/algo/vnl_brent_minimizer.h>

template <class TFloat, unsigned int VDim>
void
PointSetShootingProblem<TFloat, VDim>
::minimize_Allassonniere(const ShootingParameters &param, 
  const Matrix &q0, const Matrix &qT, Matrix &p0)
{
  unsigned int k = q0.rows();

  // Create the hamiltonian system
  HSystem hsys(q0, param.sigma, param.N, 0, param.n_threads);
  hsys.SetRalstonIntegration(param.use_ralston_method);

  // Where to store the results of the flow
  Matrix q1(k,VDim), p1(k,VDim), del_p0(k, VDim), grad_q[VDim][VDim], grad_p[VDim][VDim];
  for(unsigned int a = 0; a < VDim; a++)
    {
    for(unsigned int b = 0; b < VDim; b++)
      {
      grad_p[a][b].set_size(k,k);
      grad_q[a][b].set_size(k,k);
      }
    }

  // Transversality term and Hessian-like term
  Vector G(VDim * k);
  Matrix DG(VDim * k, VDim * k);

  // Perform optimization using the Allassonniere method
  for(unsigned int iter = 0; iter < param.iter_newton; iter++)
    {
    // Perform Hamiltonian flow
    double H = hsys.FlowHamiltonianWithGradient(p0, q1, p1, grad_q, grad_p);

    // Compute the landmark errors
    Matrix lmdiff = q1 - qT;

    // Compute the landmark part of the objective
    double fnorm = lmdiff.frobenius_norm();
    double dsq = fnorm * fnorm;

    // Fill G and Hessian
    for(unsigned int a = 0; a < VDim; a++)
      {
      for(unsigned int i = 0; i < k; i++)
        {
        // Compute the transversality vector error vector G
        G(a * k + i) = p1(i,a) + 2 * param.lambda * lmdiff(i,a);

        // Fill the Hessian-like matrix
        for(unsigned int b = 0; b < VDim; b++)
          {
          for(unsigned int j = 0; j < k; j++)
            {
            DG(a * k + i, b * k + j) = grad_p[a][b](i,j) + 2 * param.lambda * grad_q[a][b](i,j);
            }
          }
        }
      }

    // Perform singular value decomposition on the Hessian matrix, zeroing
    // out the singular values below 1.0 (TODO: use a relative threshold?)
    vnl_svd<TFloat> svd(DG, -0.001);

    int nnz = 0;
    for(int i = 0; i < svd.W().rows(); i++)
      if(svd.W()(i,i) != 0.0)
        nnz++;

    printf("SVD min: %12.8f, max: %12.8f, nnz: %d, rank: %d\n", 
      svd.sigma_min(), svd.sigma_max(), nnz, svd.rank());

    // Compute inv(DG) * G
    Vector del_p0_vec = - svd.solve(G);
    for(unsigned int a = 0; a < VDim; a++)
      for(unsigned int i = 0; i < k; i++)
        del_p0(i,a) = del_p0_vec(a * k + i);

    // Perform line search - turned out useless
    /*
    typedef PointSetShootingLineSearchCostFunction<TFloat, VDim> CostFn;
    CostFn cost_fn(param, q0, p0, qT, del_p0);

    vnl_brent_minimizer brent(cost_fn);
    brent.set_x_tolerance(0.02);
    // TFloat alpha = brent.minimize_given_bounds(0.0, 0.9, 2.0);
    */

    // Argh! What to do with alpha!
    TFloat alpha = 0.1; // 1.0 - pow( 0.9, iter + 1.0);

    // Print the current state
    printf("Iter %4d   H=%8.6f   Edist=%8.6f   E=%8.6f   |G|=%8.6f   alpha=%8.6f\n",
      iter, H, 0.5 * param.lambda * dsq, H + 0.5 * param.lambda * dsq, G.two_norm(), alpha);

    p0 += alpha * del_p0;
    }
}
/*
template <class TFloat, unsigned int VDim>
void
PointSetShootingProblem<TFloat, VDim>
::minimize_QuasiAllassonniere(const ShootingParameters &param,
  const Matrix &q0, const Matrix &qT, Matrix &p0)
{
  unsigned int k = q0.rows();

  // Create the hamiltonian system
  HSystem hsys(q0, param.sigma, param.N);

  // Where to store the results of the flow
  Matrix q1(k,VDim), p1(k,VDim), p0test(k,VDim);

  // Transversality term
  Vector G(VDim * k), Gnext(VDim * k);
  Vector dp0(VDim * k), yj(VDim * k);

  // Hessian inverse term
  Matrix Hj(VDim * k, VDim * k), Aj(VDim * k, VDim * k), Bj(VDim * k, VDim * k), Cj(VDim * k, VDim * k);
  Hj.set_identity();

  // Compute gradient G for j = 0
  double H = hsys.FlowHamiltonian(p0, q1, p1);
  for(unsigned int a = 0; a < VDim; a++)
    for(unsigned int i = 0; i < k; i++)
      G(a * k + i) = p1(i,a) + 2 * param.lambda * (q1(i,a) - qT(i,a));

  // Iterate
  TFloat alpha = 0.001;
  for(unsigned int iter = 0; iter < param.iter; iter++)
    {
    // Compute the displacement direction
    dp0 = - (Hj * G);

    // Find an alpha that satisfies Wolfe condition
    double targ = fabs(0.9 * dot_product(dp0, G));

    while(alpha > 1e-6)
      {
      // Update the current point
      for(unsigned int a = 0; a < VDim; a++)
        for(unsigned int i = 0; i < k; i++)
          p0test(i,a) = p0(i,a) + alpha * dp0(a * k + i);

      // Compute gradient at updated location
      H = hsys.FlowHamiltonian(p0test, q1, p1);
      for(unsigned int a = 0; a < VDim; a++)
        for(unsigned int i = 0; i < k; i++)
          Gnext(a * k + i) = p1(i,a) + 2 * param.lambda * (q1(i,a) - qT(i,a));

      double test = fabs(dot_product(dp0, Gnext));
      if(test < targ)
        break;
      else
        alpha = 0.5 * alpha;
      }
    if(alpha < 1e-6)
      {
      cerr << "Failed line search" << endl;
      break;
      }

    // Update p0
    p0 = p0test;

    // Compute yj - difference in gradient
    yj = Gnext - G;

    // Update the Hessian inverse matrix

    // Compute the update stuff
    Vector Z = alpha * dp0 - Hj * yj;
    double z = dot_product(Z, yj);

    for(unsigned int i = 0; i < k * VDim; i++)
      {
      for(unsigned int m = 0; m < k * VDim; m++)
        {
        // Hj(i,m) += Z(i) * Z(m) / z;
        }
      }

    // Store Gnext as G
    G = Gnext;

    // Compute the landmark part of the objective
    double fnorm = (q1 - qT).frobenius_norm();
    double dsq = fnorm * fnorm;

    // Print the current state
    printf("Iter %4d   H=%8.6f   l*Dsq=%8.6f   E=%8.6f   |G|=%8.6f\n",
      iter, H, param.lambda * dsq, H + param.lambda * dsq, G.two_norm());
    }
}
*/


#include "vnl/algo/vnl_lbfgsb.h"

template <class TFloat, unsigned int VDim>
void
PointSetShootingProblem<TFloat, VDim>
::minimize_QuasiAllassonniere(const ShootingParameters &param,
  const Matrix &q0, const Matrix &qT, Matrix &p0)
{
  // Create the minimization problem
  typedef PointSetShootingTransversalityCostFunction<TFloat, VDim> CostFn;
  CostFn cost_fn(param, q0, qT);

  // Create initial/final solution
  p0 = (qT - q0) / param.N;
  typename CostFn::DVector x = cost_fn.wide_to_tall(p0);


  // Solve the minimization problem

  vnl_lbfgsb optimizer(cost_fn);
  optimizer.set_f_tolerance(1e-9);
  optimizer.set_x_tolerance(1e-4);
  optimizer.set_g_tolerance(1e-6);
  optimizer.set_trace(true);
  optimizer.set_max_function_evals(param.iter_newton);
  optimizer.minimize(x);

  // Take the optimal solution
  p0 = cost_fn.tall_to_wide(x);
}




template <class TFloat, unsigned int VDim>
int
PointSetShootingProblem<TFloat, VDim>
::similarity_matching(const ShootingParameters &param,
    const Matrix &q0, const Matrix &qT, Matrix &q0_sim, Matrix &qT_sim,
    const Triangulation &tri_template, const Triangulation &tri_target,
    const Matrix &lab_template, const Matrix &lab_target)
{
  // Create the minimization problem
  typedef PointSetSimilarityMatchingCostFunction<TFloat, VDim> CostFn;
  CostFn cost_fn(param, q0, qT, tri_template, tri_target, lab_template, lab_target);

  // Create initial/final solution
  auto coeff_init = cost_fn.InitialSolution();
  vnl_vector<double> x(cost_fn.get_number_of_unknowns());
  cost_fn.CoeffToArray(coeff_init, x.data_block());
  vnl_random rnd;
  for(unsigned int i = 0; i < x.size(); i++)
    x[i] = x[i] + rnd.normal() * 0.01;

  // Uncomment this code to test derivative computation
  if(param.n_deriv_check > 0)
    {
    TFloat eps = 1e-6;
    vnl_vector<double> test_grad(x.size());
    double f_test;
    cost_fn.compute(x, &f_test, &test_grad);
    for(unsigned int i = 0; i < std::min((unsigned int) x.size(), param.n_deriv_check); i++)
      {
      vnl_vector<double> xtest = x;
      double f1, f2;
      xtest[i] = x[i] - eps;
      cost_fn.compute(xtest, &f1, NULL);

      xtest[i] = x[i] + eps;
      cost_fn.compute(xtest, &f2, NULL);

      printf("i = %03d,  AG = %8.4f,  NG = %8.4f\n", i, test_grad[i], (f2 - f1) / (2 * eps));
      }
    }

  // Solve the minimization problem
  cost_fn.SetVerbose(true);

  vnl_lbfgsb optimizer(cost_fn);
  optimizer.set_f_tolerance(1e-9);
  optimizer.set_x_tolerance(1e-4);
  optimizer.set_g_tolerance(1e-6);
  optimizer.set_trace(true);
  optimizer.set_max_function_evals(param.iter_grad);

  // vnl_conjugate_gradient optimizer(cost_fn);
  // optimizer.set_trace(true);
  optimizer.minimize(x);
  std::cout << "Best X: " << x << std::endl;

  // Take the optimal solution
  auto coeff_best = cost_fn.ArrayToCoeff(x.data_block());
  std::cout << "Best coeff: q = " << std::get<0>(coeff_best).r << ", " << std::get<0>(coeff_best).v << ", b = " << std::get<1>(coeff_best) << std::endl;

  // Compute the transform matrix
  vnl_matrix<TFloat> A = cost_fn.ComputeAffineMatrix(coeff_best);
  std::ofstream matrixFile;
  matrixFile.open(param.fnOutput.c_str());
  matrixFile << A;
  matrixFile.close();

  // Apply transformation to the template
  cost_fn.ComputeTransformedPoints(coeff_best, q0_sim);

  return 0;
}


template <class TFloat, unsigned int VDim>
void
PointSetShootingProblem<TFloat, VDim>
::minimize_gradient(
    const ShootingParameters &param,
    const Matrix &q0, const Matrix &qT, Matrix &p0,
    const Triangulation &tri_template, const Triangulation &tri_target,
    const Matrix &lab_template, const Matrix &lab_target)
{
  // unsigned int k = q0.rows();

  // Create the minimization problem
  typedef PointSetShootingCostFunction<TFloat, VDim> CostFn;
  CostFn cost_fn(param, q0, p0, qT, tri_template, tri_target, lab_template, lab_target);

  // Create initial/final solution
  typename CostFn::DVector x = cost_fn.wide_to_tall(p0);

  // Uncomment this code to test derivative computation
  if(param.n_deriv_check > 0)
    {
    TFloat eps = 1e-6;
    typename CostFn::DVector test_grad(x.size());
    double f_test;
    cost_fn.compute(x, &f_test, &test_grad);
    for(unsigned int i = 0; i < std::min(p0.size(), param.n_deriv_check); i++)
      {
      typename CostFn::DVector xtest = x;
      double f1, f2;
      xtest[i] = x[i] - eps;
      cost_fn.compute(xtest, &f1, NULL);

      xtest[i] = x[i] + eps;
      cost_fn.compute(xtest, &f2, NULL);

      printf("i = %03d,  AG = %8.4f,  NG = %8.4f\n", i, test_grad[i], (f2 - f1) / (2 * eps));
      }
    }

  // Solve the minimization problem
  cost_fn.SetVerbose(true);

  vnl_lbfgsb optimizer(cost_fn);
  optimizer.set_f_tolerance(1e-9);
  optimizer.set_x_tolerance(1e-4);
  optimizer.set_g_tolerance(1e-6);
  optimizer.set_trace(false);
  optimizer.set_max_function_evals(param.iter_grad);

  // vnl_conjugate_gradient optimizer(cost_fn);
  // optimizer.set_trace(true);
  optimizer.minimize(x);

  // Take the optimal solution
  p0 = cost_fn.tall_to_wide(x);
}



template <class TFloat, unsigned int VDim>
int
PointSetShootingProblem<TFloat, VDim>
::TestCurrentsAttachmentTerm(const ShootingParameters &param,
                             Matrix &q0, Matrix &qT,
                             vnl_matrix<int> &tri_template, vnl_matrix<int> &tri_target,
                             const Matrix &lab_template, const Matrix &lab_target)
{
  int m = q0.rows();
  Matrix grad_currents(m, VDim);

  TriangleCentersAndNormals <TFloat, VDim> tcan(tri_template, true);
  tcan.Forward(q0);
  cout << "TCAN test" << endl;
  cout << tcan.C.get_row(333) << endl;
  cout << tcan.N.get_row(333) << endl;
  cout << tcan.W_norm(333) << endl;

  Matrix eC(tcan.C.rows(), VDim); eC.fill(1.0);
  Matrix eN(tcan.C.rows(), VDim); eN.fill(1.0);
  Vector ea(tcan.C.rows()); ea.fill(1.0);
  Matrix eQ(tcan.C.rows(), VDim); eC.fill(1.0);
  tcan.Backward(eC, eN, ea, eQ);
  cout << eQ.get_row(333) << endl;

  typedef CurrentsAttachmentTerm<TFloat, VDim> CATerm;
  CATerm currents_attachment(
        param.attach == ShootingParameters::Current ? CATerm::CURRENTS : CATerm::VARIFOLD,
        m, qT, tri_template, tri_target, lab_template, lab_target,
        param.currents_sigma, param.n_threads);

  double value = currents_attachment.Compute(q0, &grad_currents);
  printf("Currents Attachment Value: %f\n", value);
  return 0;
}



template <class TFloat, unsigned int VDim>
int
PointSetShootingProblem<TFloat, VDim>
::minimize(const ShootingParameters &param)
{
  // Read the datasets
  vtkSmartPointer<vtkPolyData> pTemplate = ReadVTKPolyData(param.fnTemplate.c_str());
  vtkSmartPointer<vtkPolyData> pTarget = ReadVTKPolyData(param.fnTarget.c_str());

  // Read the optional control point dataset
  vtkSmartPointer<vtkPolyData> pControl = nullptr;
  if(param.fnControlMesh.length())
    pControl = ReadVTKPolyData(param.fnControlMesh.c_str());

  // Get the number of vertices and dimensionality
  if(param.attach == ShootingParameters::Euclidean)
    GreedyException::check(pTemplate->GetNumberOfPoints() == pTarget->GetNumberOfPoints(),
                           "Template and target meshes must match for the Landmark attachment term");

  // Get the number of control points
  unsigned int k = pControl ? pControl->GetNumberOfPoints() : pTemplate->GetNumberOfPoints();

  // Get the number of non-control (rider) points
  unsigned int n_riders = pControl ? pTemplate->GetNumberOfPoints() : 0;

  // Total points
  unsigned int m = k + n_riders;

  // Report number of actual vertices being used
  if(!param.do_similarity_matching)
    {
    printf("Performing geodesic shooting with %d control points and %d total landmarks.\n", k, m);
    printf("Geodesic shooting parameters: sigma = %8.4f, nt = %d, integrator = '%s'\n",
           param.sigma, param.N, param.use_ralston_method ? "Ralston": "Euler");
    }

  // Landmarks and initial momentum
  Matrix q0(m,VDim), p0(k,VDim);

  // Initialize the q0 vertex array
  for(unsigned int a = 0; a < VDim; a++)
    {
    // Control point portion of q0
    for(unsigned int i = 0; i < k; i++)
      q0(i,a) = pControl ? pControl->GetPoint(i)[a] : pTemplate->GetPoint(i)[a];

    // Rider point portion of q0
    for(unsigned int i = k; i < m; i++)
      q0(i,a) = pTemplate->GetPoint(i-k)[a];
    }

  // Initialize the target array
  Matrix qT(pTarget->GetNumberOfPoints(), VDim);
  for(unsigned int a = 0; a < VDim; a++)
    {
    for(unsigned int i = 0; i < pTarget->GetNumberOfPoints(); i++)
      {
      qT(i,a) = pTarget->GetPoint(i)[a];

      // In the simple case of Euclidean matching with no riders, initialize the momentum based on the
      // point assignment
      if(m == k && param.attach == ShootingParameters::Euclidean && param.arrInitialMomentum.length() == 0)
        p0(i,a) = (qT(i,a) - q0(i,a)) / param.N;
      }
    }

  // Read the initial momentum array
  if(param.arrInitialMomentum.length())
    {
    vtkDataArray *da_p0 = nullptr;
    if(pControl)
      da_p0 = pControl->GetPointData()->GetArray(param.arrInitialMomentum.c_str());
    else
      da_p0 = pTemplate->GetPointData()->GetArray(param.arrInitialMomentum.c_str());

    GreedyException::check(da_p0 && da_p0->GetNumberOfTuples() == k && da_p0->GetNumberOfComponents() == VDim,
          "Initial momentum array missing or has wrong dimensions");

    for(unsigned int a = 0; a < VDim; a++)
      for(unsigned int i = 0; i < k; i++)
        p0(i,a) = da_p0->GetComponent(i, a);
    }

  // Compute the triangulation of the template for non-landmark metrics
  vnl_matrix<int> tri_template, tri_target;
  Matrix lab_template(pTemplate->GetNumberOfCells(), 1, 1.0);
  Matrix lab_target(pTarget->GetNumberOfCells(), 1, 1.0);

  // Compute the triangulation of the template
  tri_template.set_size(pTemplate->GetNumberOfCells(), VDim);
  for(unsigned int i = 0; i < pTemplate->GetNumberOfCells(); i++)
    {
    if(pTemplate->GetCell(i)->GetNumberOfPoints() != VDim)
      {
      std::cerr << "Wrong number of points in template cell " << i << std::endl;
      return -1;
      }
    for(unsigned int a = 0; a < VDim; a++)
      {
      unsigned int j = pTemplate->GetCell(i)->GetPointId(a);
      tri_template(i,a) = pControl ? j + k : j;
      }
    }

  // Compute the triangulation of the target
  tri_target.set_size(pTarget->GetNumberOfCells(), VDim);
  for(unsigned int i = 0; i < pTarget->GetNumberOfCells(); i++)
    {
    if(pTarget->GetCell(i)->GetNumberOfPoints() != VDim)
      {
      std::cerr << "Wrong number of points in target cell " << i << std::endl;
      return -1;
      }
    for(unsigned int a = 0; a < VDim; a++)
      tri_target(i,a) = pTarget->GetCell(i)->GetPointId(a);
    }

  // For currents, additional configuration
  if(param.attach == ShootingParameters::Current || param.attach == ShootingParameters::Varifold)
    {
    // Read the label arrays
    if(param.arrAttachmentLabelPosteriors.length())
      {
      vtkDataArray *da_template = pTemplate->GetCellData()->GetArray(param.arrAttachmentLabelPosteriors.c_str());
      vtkDataArray *da_target = pTarget->GetCellData()->GetArray(param.arrAttachmentLabelPosteriors.c_str());
      GreedyException::check(da_template && da_target && da_template->GetNumberOfComponents() == da_target->GetNumberOfComponents(),
            "Label posterior arrays in template and target missing or do not match");
      int n_labels = da_template->GetNumberOfComponents();
      lab_template.set_size(tri_template.rows(), n_labels);
      for(unsigned int i = 0; i < tri_template.rows(); i++)
        for(unsigned int l = 0; l < n_labels; l++)
          lab_template(i,l) = da_template->GetComponent(i,l);
      lab_target.set_size(tri_target.rows(), n_labels);
      for(unsigned int i = 0; i < tri_target.rows(); i++)
        for(unsigned int l = 0; l < n_labels; l++)
          lab_target(i,l) = da_target->GetComponent(i,l);
      }

    if(param.test_currents_attachment)
      {
      TestCurrentsAttachmentTerm(param, q0, qT, tri_template, tri_target, lab_template, lab_target);
      return 0;
      }
    }

  // Are we doing similarity matching - then it's a whole separate thing
  if(param.do_similarity_matching)
    {
    // Perform similarity matching
    Matrix q_fit(m,VDim);
    Matrix q_fit_inv(m,VDim);
    int rc = similarity_matching(param, q0, qT, q_fit, q_fit_inv, tri_template, tri_target, lab_template, lab_target);

    // This code will save transformed mesh
    /*
    if(rc == 0)
      {
      vtkNew<vtkPolyData> pTran; pTran->DeepCopy(pTemplate);
      for(unsigned int i = 0; i < m; i++)
        pTran->GetPoints()->SetPoint(i, q_fit.get_row(i).data_block());
      WriteVTKData(pTran, "test_fwd.vtk");
      vtkNew<vtkPolyData> pTranTarg; pTranTarg->DeepCopy(pTemplate);
      for(unsigned int i = 0; i < m; i++)
        pTranTarg->GetPoints()->SetPoint(i, q_fit_inv.get_row(i).data_block());
      WriteVTKData(pTranTarg, "test_inv.vtk");
      }
    */
    return rc;
    }

  // Run some iterations of gradient descent
  if(param.iter_grad > 0)
    {
    minimize_gradient(param, q0, qT, p0, tri_template, tri_target, lab_template, lab_target);
    }

  if(param.iter_newton > 0)
    {
    minimize_Allassonniere(param, q0, qT, p0);
    }

  // Which polydata are we saving?
  vtkPolyData *pResult = pControl ? pControl : pTemplate;

  // Genererate the output momentum map
  vtkDoubleArray *arr_p = vtkDoubleArray::New();
  arr_p->SetNumberOfComponents(VDim);
  arr_p->SetNumberOfTuples(k);
  arr_p->SetName("InitialMomentum");

  for(unsigned int a = 0; a < VDim; a++)
    arr_p->FillComponent(a, 0);

  for(unsigned int a = 0; a < VDim; a++)
    for(unsigned int i = 0; i < k; i++)
      arr_p->SetComponent(i, a, p0(i,a));

  // Set the momenta
  pResult->GetPointData()->AddArray(arr_p);

  // Store the shooting parameters as field data
  vtk_set_scalar_field_data(pResult, "lddmm_nt", param.N);
  vtk_set_scalar_field_data(pResult, "lddmm_sigma", param.sigma);
  vtk_set_scalar_field_data(pResult, "lddmm_ralston", param.use_ralston_method ? 1.0 : 0.0);

  // Save the mesh
  WriteMesh(pResult, param.fnOutput.c_str());

  // If saving paths requested
  if(param.fnOutputPaths.size())
    {
    // Create and flow a system
    HSystem hsys(q0, param.sigma, param.N, m - k, param.n_threads);
    hsys.SetRalstonIntegration(param.use_ralston_method);
    Matrix q1, p1;
    hsys.FlowHamiltonian(p0, q1, p1);

    // Get the number of points in the output mesh
    unsigned int nv = pTemplate->GetNumberOfPoints();

    // Apply the flow to the points in the rest of the mesh
    vtkDoubleArray *arr_v = vtkDoubleArray::New();
    arr_v->SetNumberOfComponents(VDim);
    arr_v->SetNumberOfTuples(nv);
    arr_v->SetName("Velocity");
    pTemplate->GetPointData()->AddArray(arr_v);

    /*
    vtkDoubleArray *arr_p = vtkDoubleArray::New();
    arr_p->SetNumberOfComponents(VDim);
    arr_p->SetNumberOfTuples(np);
    arr_p->SetName("Momentum");
    pTemplate->GetPointData()->AddArray(arr_p);
    */

    // Apply Euler method to the mesh points
    double dt = hsys.GetDeltaT();
    for(unsigned int t = 1; t < param.N; t++)
      {
      for(unsigned int i = 0; i < nv; i++)
        {
        TFloat qi[VDim], vi[VDim];

        for(unsigned int a = 0; a < VDim; a++)
          qi[a] = pTemplate->GetPoint(i)[a];

        // Interpolate the velocity at each mesh point
        hsys.InterpolateVelocity(t-1, qi, vi);

        // Update the position using Euler's method
        for(unsigned int a = 0; a < VDim; a++)
          qi[a] += dt * vi[a];

        for(unsigned int a = 0; a < VDim; a++)
          pTemplate->GetPoints()->SetPoint(i, qi);

        for(unsigned int a = 0; a < VDim; a++)
          {
          arr_v->SetComponent(i, a, vi[a]);
          // arr_p->SetComponent(i, a, hsys.GetPt(t)(i,a));
          }
        }

      // Output the intermediate mesh
      char buffer[1024];
      snprintf(buffer, 1024, param.fnOutputPaths.c_str(), t);
      WriteMesh(pTemplate, buffer);
      }
    }

  return 0;
}

ShootingParameters lmshoot_parse_commandline(CommandLineHelper &cl, bool parse_template_params)
{
  ShootingParameters param;

  // Process parameters
  while(!cl.is_at_end())
  {
    // Read the next command
    std::string arg = cl.read_command();

    if(arg == "-m")
    {
      param.fnTemplate = cl.read_existing_filename();
      param.fnTarget = cl.read_existing_filename();
    }
    else if(arg == "-c")
    {
      param.fnControlMesh = cl.read_existing_filename();
    }
    else if(arg == "-G")
    {
      param.do_similarity_matching = true;
    }
    else if(arg == "-o")
    {
      param.fnOutput = cl.read_output_filename();
    }
    else if(arg == "-O")
    {
      param.fnOutputPaths = cl.read_string();
    }
    else if(arg == "-s")
    {
      param.sigma = cl.read_double();
    }
    else if(arg == "-l")
    {
      param.lambda = cl.read_double();
    }
    else if(arg == "-g")
    {
      param.gamma = cl.read_double();
    }
    else if(arg == "-n")
    {
      param.N = (unsigned int) cl.read_integer();
    }
    else if(arg == "-R")
    {
      param.use_ralston_method = true;
    }
    else if(arg == "-d")
    {
      param.dim = (unsigned int) cl.read_integer();
    }
    else if(arg == "-i")
    {
      param.iter_grad = (unsigned int) cl.read_integer();
      param.iter_newton = (unsigned int) cl.read_integer();
    }
    else if(arg == "-C")
    {
      param.constrained_mu_init = cl.read_double();
      param.constrained_mu_mult = cl.read_double();
    }
    else if(arg == "-f")
    {
      param.use_float = true;
    }
    else if(arg == "-p")
    {
      param.arrInitialMomentum = cl.read_string();
    }
    else if(arg == "-L")
    {
      param.arrAttachmentLabelPosteriors = cl.read_string();
    }
    else if(arg == "-J")
    {
      param.w_jacobian = cl.read_double();
    }
    else if(arg == "-t")
    {
      param.n_threads = cl.read_integer();
    }
    else if(arg == "-D")
    {
      param.n_deriv_check = cl.read_integer();
    }
    else if(arg == "-a")
    {
      std::string mode = cl.read_string();
      if(mode == "L")
        param.attach = ShootingParameters::Euclidean;
      else if(mode == "C")
        param.attach = ShootingParameters::Current;
      else if(mode == "V")
        param.attach = ShootingParameters::Varifold;
      else
      {
        throw GreedyException("Unknown attachment type %s", mode.c_str());
      }
    }
    else if(arg == "-S")
    {
      param.currents_sigma = cl.read_double();
    }
    else if(arg == "-test-currents")
    {
      param.test_currents_attachment = true;
    }
    else if(arg == "-h")
    {
      lmshoot_usage(parse_template_params);
    }
    else
    {
      throw GreedyException("Unknown option: %s", arg.c_str());
    }
  }

  // Check parameters
  if(!param.do_similarity_matching)
  {
    GreedyException::check(param.sigma > 0, "Missing or negative sigma parameter");
    GreedyException::check(param.N > 0 && param.N < 10000, "Incorrect N parameter");
  }
  GreedyException::check(param.attach == ShootingParameters::Euclidean || param.currents_sigma > 0,
        "Missing sigma parameter for current/varifold metric");
  if(parse_template_params)
    GreedyException::check(param.dim >= 2 && param.dim <= 3, "Incorrect N parameter");
  GreedyException::check(param.fnTemplate.length(), "Missing template filename");
  GreedyException::check(param.fnTarget.length(), "Missing target filename");
  GreedyException::check(param.fnOutput.length(), "Missing output filename");

  // Set the number of threads if not specified
  if(param.n_threads == 0)
    param.n_threads = std::thread::hardware_concurrency();
  else
    // Set the threads in ITK
    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(param.n_threads);

  // Specialize by dimension
  return param;
}

template class PointSetShootingProblem<float, 2>;
template class PointSetShootingProblem<float, 3>;
template class PointSetShootingProblem<double, 2>;
template class PointSetShootingProblem<double, 3>;
