#include "PointSetGeodesicToWarp.h"
#include "GreedyMeshIO.h"
#include "GreedyException.h"
#include "PointSetHamiltonianSystem.h"
#include "lddmm_data.h"
#include <iostream>
#include "VTKArrays.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "FastLinearInterpolator.h"
#include "itkMultiThreaderBase.h"
#include "CommandLineHelper.h"

#include <cstdarg>

using namespace std;

int lmtowarp_usage(bool print_template_param)
{
  cout << "lmtowarp: Generate warp field from landmark geodesic shooting" << endl;
  cout << "Usage:" << endl;
  cout << "  lmtowarp [options]" << endl;
  cout << "Required options:" << endl;
  cout << "  -m mesh.vtk        : Mesh with the InitialMomentum array defined" << endl;
  cout << "Warp generation mode:" << endl;
  cout << "  -r image.nii       : Reference image (defines warp space)" << endl;
  cout << "  -o warp.nii        : Output deformation field" << endl;
  cout << "Mesh warping mode:" << endl;
  cout << "  -M in.vtk out.vtk  : additional meshes to apply warp to" << endl;
  cout << "Additional options:" << endl;
  if(print_template_param)
    cout << "  -d dim             : problem dimension (3)" << endl;
  cout << "  -n N               : number of time steps (default: read from mesh.vtk)" << endl;
  cout << "  -s sigma           : kernel standard deviation (default: read from mesh.vtk)" << endl;
  cout << "  -R                 : use Ralston integration (default: read from mesh.vtk)" << endl;
  cout << "  -g mask.nii        : limit warp computation to a masked region" << endl;
  cout << "  -B                 : use brute force warp computation (not approximation)" << endl;
  cout << "  -t n_threads       : limit number of concurrent threads to n_threads" << endl;
  cout << "Animation:" << endl;
  cout << "  -a k               : save an animation frame every k-th timepoint" << endl;
  cout << "                       output files must have %03d pattern in them" << endl;

  return -1;
}

template <class TPixel, unsigned int VDim>
void
PointSetGeodesicToWarp<TPixel, VDim>
::UpdateAndWriteMesh(
  vtkPolyData *mesh, const Matrix &x, const Matrix &v, const Matrix &x0,
  const std::string filePattern,
  int k)
{
  // Velocity array
  vtkDoubleArray *arr_v = vtkDoubleArray::New();
  arr_v->SetNumberOfComponents(VDim);
  arr_v->SetNumberOfTuples(v.rows());
  arr_v->SetName("Velocity");
  mesh->GetPointData()->AddArray(arr_v);

  // Initial position array
  vtkDoubleArray *arr_x0 = vtkDoubleArray::New();
  arr_x0->SetNumberOfComponents(VDim);
  arr_x0->SetNumberOfTuples(v.rows());
  arr_x0->SetName("InitialPosition");
  mesh->GetPointData()->AddArray(arr_x0);

  // Assign new points to the mesh
  for(int i = 0; i < x.rows(); i++)
    {
    double x_out[VDim];
    for(int a = 0; a < VDim; a++) 
      {
      x_out[a] = x(i,a);
      arr_v->SetComponent(i,a,v(i,a));
      arr_x0->SetComponent(i,a,x0(i,a));
      }
    mesh->GetPoints()->SetPoint(i,x_out);
    }

  // Write the mesh
  char buffer[2048];
  snprintf(buffer, 2048, filePattern.c_str(), k);
  WriteMesh(mesh, buffer);
}

template <class TPixel, unsigned int VDim>
typename PointSetGeodesicToWarp<TPixel, VDim>::VectorImagePointer
PointSetGeodesicToWarp<TPixel, VDim>
::brute_force_method(
    const WarpGenerationParameters &param,
    HSystem &hsys, int tdir, int tStart, int tEnd)
{
  // Read the reference image
  ImagePointer imRef = LDDMM::img_read(param.fnReference.c_str());

  // Read the optional mask image. If not supplied, create a mask of all ones
  // to simplify the code below (fewer ifs)
  ImagePointer imMask;
  if(param.fnMask.size())
    {
    imMask = LDDMM::img_read(param.fnMask.c_str());
    itkAssertOrThrowMacro(imMask->GetBufferedRegion() == imRef->GetBufferedRegion(),
                          "Region mismatch between mask and reference");
    }
  else
    {
    imMask = LDDMM::new_img(imRef);
    imMask->FillBuffer(1.0);
    }

  // Count the number of masked pixels
  unsigned int nv_masked = 0;
  typedef itk::ImageRegionIteratorWithIndex<ImageType> IterType;
  for(IterType it(imMask, imMask->GetBufferedRegion()); !it.IsAtEnd(); ++it)
    nv_masked += it.Value() >= 1.0 ? 1 : 0;

  // Create an array to store the RAS coordinates of the points
  vnl_matrix<TPixel> X(nv_masked, VDim);

  // Initialize all the points inside the mask
  unsigned int ix = 0;
  itk::Point<double, VDim> pt_lps;
  for(IterType it(imMask, imMask->GetBufferedRegion()); !it.IsAtEnd(); ++it)
    {
    if(it.Value() >= 1.0)
      {
      // Compute the RAS coordinate of the current vertex position (applying displacement phi)
      imMask->TransformIndexToPhysicalPoint(it.GetIndex(), pt_lps);
      for(unsigned int a = 0; a < VDim; a++)
        X(ix,a) = a < 2 ? -pt_lps[a] : pt_lps[a];
      ++ix;
      }
    }

  // F for the Gaussian
  double gaussian_f = -1.0 / (2 * param.sigma * param.sigma);

  // Cutoff where the Gaussian is < 1e-6
  double d2_cutoff = 27.63102 * param.sigma * param.sigma;

  // Delta-t
  double dt = tdir * 1.0 / (param.N - 1);

  // Perform iteration over time
  for(int t = tStart; t != tEnd; t+=tdir)
    {
    // Get all landmarks
    const Matrix &qt = hsys.GetQt(t);
    const Matrix &pt = hsys.GetPt(t);

    // Multithread the vertices
    itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
    itk::ImageRegion<1> full_region({{0}}, {{nv_masked}});
    mt->ParallelizeImageRegion<1>(
          full_region,
          [&qt, &pt, &X, dt, d2_cutoff, gaussian_f](const itk::ImageRegion<1> &thread_region)
      {
        int i0 = thread_region.GetIndex(0);
        int i1 = i0 + thread_region.GetSize(0);
        for(int i = i0; i < i1; i++)
          {
          // Velocity at this index
          TPixel xi[VDim], vi[VDim];
          for(unsigned int a = 0; a < VDim; a++)
            {
            xi[a] = X(i,a);
            vi[a] = 0.0;
            }

          // Loop over all q nodes
          for(unsigned int j = 0; j < qt.rows(); j++)
            {
            // Compute distance to that point
            double delta, d2 = 0;
            for(unsigned int a = 0; a < VDim; a++)
              {
              delta = xi[a] - qt(j,a);
              d2 += delta * delta;
              }

            // Only proceed if distance is below cutoff
            if(d2 < d2_cutoff)
              {
              // Take the exponent
              double g = exp(gaussian_f * d2);

              // Scale momentum by exponent
              for(unsigned int a = 0; a < VDim; a++)
                vi[a] += g * pt(j,a);
              }
            }

          // Use Euler's method to get position at next timepoint, encoding it as a LPS
          // displacement
          for(unsigned int a = 0; a < VDim; a++)
            X(i,a) += vi[a] * dt;
          }
      }, nullptr);

    std::cout << "." << std::flush;
    }

  // Form the phi image
  VectorImagePointer imPhi = LDDMM::new_vimg(imRef);
  ix = 0;
  IterType it(imMask, imMask->GetBufferedRegion());
  itk::ImageRegionIteratorWithIndex<VectorImageType> itPhi(imPhi, imMask->GetBufferedRegion());
  for(; !it.IsAtEnd(); ++it,++itPhi)
    {
    if(it.Value() >= 1.0)
      {
      // Compute the RAS coordinate of the current vertex position (applying displacement phi)
      imMask->TransformIndexToPhysicalPoint(it.GetIndex(), pt_lps);
      for(unsigned int a = 0; a < VDim; a++)
        {
        double x_lps = a < 2 ? -X(ix,a) : X(ix,a);
        double phi_lps = x_lps - pt_lps[a];
        itPhi.Value()[a] = phi_lps;
        }
      ++ix;
      }
    }

  return imPhi;
}

template <class TPixel, unsigned int VDim>
int
PointSetGeodesicToWarp<TPixel, VDim>
::run(const WarpGenerationParameters &param)
{
  // Read the VTK mesh containing the points
  vtkSmartPointer<vtkPolyData> mesh = ReadVTKPolyData(param.fnMesh.c_str());

  // Read the momentum field
  vtkDataArray *arr_p0 = mesh->GetPointData()->GetArray("InitialMomentum");
  if(!arr_p0 || arr_p0->GetNumberOfComponents() != VDim)
    throw GreedyException("Failed to read initial momentum from %s", param.fnMesh.c_str());

  // Read parameters from mesh if they were not supplied by the user
  double sigma = param.sigma > 0.0 ? param.sigma
                                   : vtk_get_scalar_field_data(mesh, "lddmm_sigma", 0.0);
  int N = param.N > 0.0 ? param.N
                        : vtk_get_scalar_field_data(mesh, "lddmm_nt", 0);
  bool use_ralston_method = param.use_ralston_method ? true
                                                     : vtk_get_scalar_field_data(mesh, "lddmm_ralston", false);

  GreedyException::check(sigma > 0, "Missing or negative sigma parameter");
  GreedyException::check(N > 0 && param.N < 10000, "Incorrect N parameter");

  // Count the number of non-null entries
  vector<unsigned int> index;
  for(unsigned int i = 0; i < arr_p0->GetNumberOfTuples(); i++)
    {
    bool has_value = 1;
    for(unsigned int a = 0; a < VDim; a++)
      {
      if(std::isnan(arr_p0->GetComponent(i,a)))
        {
        has_value = 0;
        break;
        }
      }
    if(has_value)
      index.push_back(i);
    }

  // Populate the q0 and p0 arrays
  unsigned int k = index.size();
  Matrix q0(k, VDim), p0(k,VDim), q1(k, VDim), p1(k,VDim);

  for(unsigned int i = 0; i < k; i++)
    {
    for(unsigned int a = 0; a < VDim; a++)
      {
      q0(i,a) = mesh->GetPoint(index[i])[a];
      p0(i,a) = arr_p0->GetComponent(i,a);
      }
    }

  // We have read the mesh successfully
  // Create the hamiltonian system
  HSystem hsys(q0, sigma, N, 0, param.n_threads);
  hsys.SetRalstonIntegration(use_ralston_method);

  // Describe the parameters
  printf("Geodesic shooting parameters: sigma = %8.4f, nt = %d, integrator = '%s'\n",
         sigma, N, use_ralston_method ? "Ralston": "Euler");

  // Flow without gradients - we have streamlines
  hsys.FlowHamiltonian(p0, q1, p1);

  // Direction of interpolation
  int tdir = 1;
  int tStart = (tdir > 0) ? 0 : N - 1;
  int tEnd = (tdir > 0) ? N : 0;

  // F for the Gaussian
  double gaussian_f = -1.0 / (2 * sigma * sigma);

  // Cutoff where the Gaussian is < 1e-6
  double d2_cutoff = 27.63102 * sigma * sigma;

  // Delta-t
  double dt = tdir * 1.0 / (N - 1);

  // Apply the warp to other meshes
  std::list<WarpGenerationParameters::MeshPair>::const_iterator it;
  for(it = param.fnWarpMeshes.begin(); it != param.fnWarpMeshes.end(); ++it)
    {
    // Read the VTK mesh
    vtkSmartPointer<vtkPolyData> mesh_to_warp = ReadVTKPolyData(it->first.c_str());

    // Create the arrays to track the point positions
    Matrix m_x(mesh_to_warp->GetNumberOfPoints(), VDim);

    // Initialize
    for(int i = 0; i < m_x.rows(); i++)
      for(int a = 0; a < VDim; a++)
        m_x(i,a) = mesh_to_warp->GetPoint(i)[a];

    // Store the initial array
    Matrix m_x0 = m_x;

    // Some print-out
    std::cout << "Warping mesh " << it->first << " : " << std::flush;

    // Iterate over time
    for(int t = tStart; t != tEnd; t+=tdir)
      {
      // Get all landmarks
      const Matrix &qt = hsys.GetQt(t);
      const Matrix &pt = hsys.GetPt(t);

      // Create a vector for the velocity
      Matrix v(m_x.rows(), VDim);

      // Compute the velocity for each landmark
      for(int i = 0; i < m_x.rows(); i++)
        {
        // Local arrays for speed
        double xi[VDim];
        for(int a = 0; a < VDim; a++) 
          {
          xi[a] = m_x(i,a);
          v(i,a) = 0;
          }

        // Loop over all q nodes
        for(int j = 0; j < qt.rows(); j++)
          {
          // Compute distance to that point
          double delta, d2 = 0;
          for(int a = 0; a < VDim; a++) 
            {
            delta = xi[a] - qt(j,a);
            d2 += delta * delta;
            }

          // Only proceed if distance is below cutoff
          if(d2 < d2_cutoff)
            {
            // Take the exponent
            double g = exp(gaussian_f * d2);

            // Scale momentum by exponent
            for(int a = 0; a < VDim; a++) 
              v(i,a) += g * pt(j,a);
            }
          }

        // Use Euler's method to get position at next timepoint
        for(int a = 0; a < VDim; a++) 
          m_x(i,a) += v(i,a) * dt;
        }

      // If this is an animation frame, save it
      if((param.anim_freq > 0 && 0 == (t + 1) % param.anim_freq) || t + 1 == tEnd)
        UpdateAndWriteMesh(mesh_to_warp, m_x, v, m_x0, it->second, t + 1);

      std::cout << "." << std::flush;
      }

    // Done for this mesh - just have to save it
    std::cout << std::endl;
    }

  // Compute warp if that was requested
  if(param.fnReference.size() && param.fnOutWarp.size())
    {
    if(param.brute)
      {
      VectorImagePointer imPhi = brute_force_method(param, hsys, tdir, tStart, tEnd);
      LDDMM::vimg_write(imPhi, param.fnOutWarp.c_str());
      }
    else
      {
      // Read the reference image
      ImagePointer imRef;
      LDDMM::img_read(param.fnReference.c_str(), imRef);

      // If mask specified, read the mask
      ImagePointer imMask;
      if(param.fnMask.size())
        imMask = LDDMM::img_read(param.fnMask.c_str());

      // Image for splatting
      VectorImagePointer
          imSplat = LDDMM::new_vimg(imRef),
          imVelocity = LDDMM::new_vimg(imRef),
          imLagragean = LDDMM::new_vimg(imRef),
          imPhi = LDDMM::new_vimg(imRef);

      // Compute the scaling factor for the Gaussian smoothing
      double scaling = 1.0;
      for(unsigned int a = 0; a < VDim; a++)
        {
        scaling *= sqrt(2 * vnl_math::pi) * sigma / imRef->GetSpacing()[a];
        }

      // Iterate over time
      for(int t = tStart; t != tEnd; t+=tdir)
        {
        // Get all landmarks
        const Matrix &qt = hsys.GetQt(t);
        const Matrix &pt = hsys.GetPt(t);

        // Splatting approach
        // Splat each landmark onto the splat image
        imSplat->FillBuffer(typename VectorImageType::PixelType(0.0));

        // Create an interpolator for splatting
        typedef FastLinearInterpolator<VectorImageType,TPixel,VDim> FastInterpolator;
        FastInterpolator flint(imSplat);

        for(int i = 0; i < k; i++)
          {
          // Map landmark point to continuous index
          ContIndexType cix;
          PointType point;
          typename VectorImageType::InternalPixelType vec;
          for(unsigned int a = 0; a < VDim; a++)
            {
            // Here we have to correct for the fact that the landmark coordinates are
            // assumed to be in RAS space, but point is in LPS space. We therefore have
            // to apply the LPS/RAS transform before splatting
            if(a < 2)
              {
              point[a] = -qt(i,a);
              vec[a] = -pt(i,a);
              }
            else
              {
              point[a] = qt(i,a);
              vec[a] = pt(i,a);
              }

            }
          imRef->TransformPhysicalPointToContinuousIndex(point, cix);

          // Splat landmark's momentum at the point location
          flint.Splat(cix.GetVnlVector().data_block(), &vec);
          }

        // Now all the momenta have been splatted. The next step is to smooth with a Gaussian
        typename LDDMM::Vec vec_sigma; vec_sigma.Fill(sigma);
        LDDMM::vimg_smooth(imSplat, imVelocity, vec_sigma);

        // Accumulate the velocity into phi - using imSplat as a temporary
        LDDMM::vimg_scale_in_place(imVelocity, scaling);

        // Euler interpolation
        LDDMM::interp_vimg(imVelocity, imPhi, 1.0, imSplat, false, true);
        LDDMM::vimg_add_scaled_in_place(imPhi, imSplat, tdir * 1.0 / (N - 1));

        cout << "." << flush;

        // Save the warp if needed
        if(t + 1 == tEnd || (param.anim_freq && 0 == (t + 1) % param.anim_freq))
          {
          char buffer[2048];
          snprintf(buffer, 2048, param.fnOutWarp.c_str(), t + 1);

          // Save the warp
          LDDMM::vimg_write(imPhi, buffer);
          }
        }
      }
    }

  return 0;
}


WarpGenerationParameters
lmtowarp_parse_commandline(CommandLineHelper &cl, bool parse_template_params)
{
  WarpGenerationParameters param;

  // Process parameters
  while(!cl.is_at_end())
  {
    string arg = cl.read_command();
    if(arg == "-r")
    {
      param.fnReference = cl.read_existing_filename();
    }
    else if(arg == "-m")
    {
      param.fnMesh = cl.read_existing_filename();
    }
    else if(arg == "-o")
    {
      param.fnOutWarp = cl.read_output_filename();
    }
    else if(arg == "-g")
    {
      param.fnMask = cl.read_existing_filename();
    }
    else if(arg == "-s")
    {
      param.sigma = cl.read_double();
    }
    else if(arg == "-d")
    {
      param.dim = cl.read_integer();
    }
    else if(arg == "-f")
    {
      param.use_float = true;
    }
    else if(arg == "-n")
    {
      param.N = cl.read_integer();
    }
    else if(arg == "-R")
    {
      param.use_ralston_method = true;
    }
    else if(arg == "-a")
    {
      param.anim_freq = cl.read_integer();
    }
    else if(arg == "-B")
    {
      param.brute = true;
    }
    else if(arg == "-t")
    {
      param.n_threads = cl.read_integer();
    }
    else if(arg == "-M")
    {
      string fn1 = cl.read_existing_filename();
      string fn2 = cl.read_output_filename();
      WarpGenerationParameters::MeshPair pair = std::make_pair(fn1,fn2);
      param.fnWarpMeshes.push_back(pair);
    }
    else if(arg == "-h")
    {
      lmtowarp_usage(parse_template_params);
    }
    else
    {
      throw GreedyException("Unknown option %s", arg.c_str());
    }
  }

  if(parse_template_params)
    GreedyException::check(param.dim >= 2 && param.dim <= 3, "Incorrect N parameter");

  // Set the number of threads if not specified
  if(param.n_threads == 0)
    param.n_threads = std::thread::hardware_concurrency();
  else
    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(param.n_threads);

  return param;
}

template class PointSetGeodesicToWarp<float, 2>;
template class PointSetGeodesicToWarp<float, 3>;
template class PointSetGeodesicToWarp<double, 2>;
template class PointSetGeodesicToWarp<double, 3>;

