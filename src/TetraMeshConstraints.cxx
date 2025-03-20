#include "TetraMeshConstraints.h"
#include <vtkUnstructuredGrid.h>
#include "GreedyException.h"
#include "AffineTransformUtilities.h"
#include "FastLinearInterpolator.h"
#include <vnl/algo/vnl_matrix_inverse.h>
#include <vnl/vnl_cross.h>
#include <random>
#include <vtkHexahedron.h>
#include "GreedyMeshIO.h"
#include <vtkSubdivideTetra.h>
#include "DifferentiableScalingAndSquaring.h"

// Tetra/triangle volume computations
template <unsigned int VDim>
inline double simplex_volume(const vnl_matrix<double> &, const vnl_vector<int> &, vnl_matrix_fixed<double, VDim+1, VDim> * = nullptr)
{
  return 0;
}

template <>
inline double simplex_volume<3u>(const vnl_matrix<double> &x, const vnl_vector<int> &vindex, vnl_matrix_fixed<double, 4, 3> *dx)
{
  vnl_vector_fixed<double, 3> A, B, C;
  const double *X0 = x[vindex[0]];
  const double *X1 = x[vindex[1]];
  const double *X2 = x[vindex[2]];
  const double *X3 = x[vindex[3]];
  for(unsigned int j = 0; j < 3; j++)
    {
      A[j] = X1[j] - X0[j];
      B[j] = X2[j] - X0[j];
      C[j] = X3[j] - X0[j];
    }
  auto AxB = vnl_cross_3d(A, B);
  double v = dot_product(AxB, C);
  if(dx)
    {
      auto CxA = vnl_cross_3d(C, A);
      auto BxC = vnl_cross_3d(B, C);
      dx->set_row(0, -(AxB + CxA + BxC));
      dx->set_row(1, BxC);
      dx->set_row(2, CxA);
      dx->set_row(3, AxB);
    }

  return v;
}

template <unsigned int VDim>
void
    TetraVolumeLayer<VDim>
    ::SetIndex(const vnl_vector<int> v_index)
{
  this->v_index = v_index;
}

template <unsigned int VDim>
double
    TetraVolumeLayer<VDim>
    ::Forward(const vnl_matrix<double> &x, bool need_grad)
{
  return simplex_volume<VDim>(x, v_index, need_grad ? &dv_dx : nullptr);
}

template <unsigned int VDim>
void
    TetraVolumeLayer<VDim>
    ::Backward(double d_tetra_vol, vnl_matrix<double> &d_x)
{
  for(unsigned int k = 0; k < VDim + 1; k++)
    {
      double *p = d_x[v_index[k]];
      for(unsigned int i = 0; i < VDim; i++)
        p[i] += dv_dx[k][i] * d_tetra_vol;
    }
}


template<class TFloat, unsigned int VDim>
TetraMeshConstraints<TFloat, VDim>
    ::TetraMeshConstraints()
{

}

template<class TFloat, unsigned int VDim>
void TetraMeshConstraints<TFloat, VDim>
    ::SetMesh(vtkUnstructuredGrid *mesh)
{
  m_MeshVTK = mesh;

  // Allocate and assign the vertex coordinates
  m_TetraX_RAS.set_size(mesh->GetNumberOfPoints(), VDim);
  m_TetraX_Vox.set_size(mesh->GetNumberOfPoints(), VDim);
  for(unsigned int i = 0; i < mesh->GetNumberOfPoints(); i++)
    {
      double *x = mesh->GetPoint(i);
      for(unsigned int j = 0; j < VDim; j++)
        m_TetraX_RAS(i,j) = x[j];
    }

  // For neighbor search
  auto nbr = vtkNew<vtkIdList>();

  // Allocate and assign the tetrahedral vertex indices
  m_TetraVI.set_size(mesh->GetNumberOfCells(), VDim + 1);
  m_TetraNbr.clear();
  m_TetraVolumeLayer.resize(mesh->GetNumberOfCells());
  for(unsigned int i = 0; i < mesh->GetNumberOfCells(); i++)
    {
      auto *cell = mesh->GetCell(i);
      if(cell->GetNumberOfPoints() != VDim + 1)
        throw GreedyException("Mesh has cells of incorrect dimension");

      // Assign vertices to the tetrahedra
      for(unsigned int j = 0; j < VDim+1; j++)
        m_TetraVI(i,j) = cell->GetPointId(j);

      // Check the tetrahedral orientation - volumes in the input tetrahedron
      // should all be positive, if negative, flip the order
      double vol = simplex_volume<VDim>(m_TetraX_RAS, m_TetraVI.get_row(i), nullptr);
      if(vol < 0)
        {
          auto v0 = m_TetraVI(i, 0);
          m_TetraVI(i, 0) = m_TetraVI(i, 1);
          m_TetraVI(i, 1) = v0;
        }

      vol = simplex_volume<VDim>(m_TetraX_RAS, m_TetraVI.get_row(i), nullptr);
      if(vol < 0)
        std::cout << "Something is wrong" << std::endl;

      // Initialize the volume layer
      m_TetraVolumeLayer[i].SetIndex(m_TetraVI.get_row(i));

      // Find neighbors across each edge
      vtkIdType edge[VDim];
      for(unsigned int j = 0; j < VDim+1; j++)
        {
          vtkIdType *p_edge = edge;
          for(unsigned int k = 0; k < VDim+1; k++)
            if(k != j)
              *p_edge++ = m_TetraVI(i,k);

          nbr->Reset();
          mesh->GetCellNeighbors(i, VDim, edge, nbr);
          if(nbr->GetNumberOfIds() > 1)
            throw GreedyException("Cell %d has wrong number of neighbors across %d's face: %d", i, j, nbr->GetNumberOfIds());

          if(nbr->GetNumberOfIds() == 1 && nbr->GetId(0) > i)
            m_TetraNbr.push_back(std::make_pair(i, nbr->GetId(0)));
        }
    }

  // Allocate the internal computation arrays
  m_TetraX_RAS_Disp.set_size(m_MeshVTK->GetNumberOfPoints(), VDim);
  m_TetraX_RAS_Warped.set_size(m_MeshVTK->GetNumberOfPoints(), VDim);
  m_D_TetraX_RAS_Warped.set_size(m_MeshVTK->GetNumberOfPoints(), VDim);
  m_TetraVol.set_size(m_TetraVI.rows());
  m_TetraVol_Warped.set_size(m_TetraVI.rows());
  m_D_TetraVol_Warped.set_size(m_TetraVI.rows());
}

template<class TFloat, unsigned int VDim>
void TetraMeshConstraints<TFloat, VDim>
    ::SetReferenceImage(ImageBaseType *ref_space)
{
  if(!m_MeshVTK)
    throw GreedyException("TetraMeshConstraints::SetReferenceImage called before SetMesh");

  m_Reference = ref_space;

  // Extract transforms from the reference space
  GetVoxelSpaceToNiftiSpaceTransform(ref_space, A_vox_to_ras, b_vox_to_ras);

  // Compute the inverse transforms
  A_ras_to_vox = vnl_matrix_inverse<double>(A_vox_to_ras.as_matrix()).as_matrix();
  b_ras_to_vox = - A_ras_to_vox * b_vox_to_ras;

  // Compute the voxel-space coordinates of the mesh
  for(unsigned int i = 0; i < m_MeshVTK->GetNumberOfPoints(); i++)
    m_TetraX_Vox.set_row(i, A_ras_to_vox * m_TetraX_RAS.get_row(i) + b_ras_to_vox);

  // Compute the baseline tetrahedral volumes
  for(unsigned k = 0; k < m_TetraVI.rows(); k++)
    m_TetraVol[k] = m_TetraVolumeLayer[k].Forward(m_TetraX_RAS, false);

  // TODO: for each vertex, compute the sparse matrix of interpolation weights
  // and indices so we don't have to call interpolator.Splat at every evaluation

}


template<class TFloat, unsigned int VDim>
double
    TetraMeshConstraints<TFloat, VDim>
    ::ComputeObjectiveAndGradientDisp(
        const vnl_matrix<double> &disp_ras, vnl_matrix<double> &grad, double weight)
{
  // Number of vertices and tetrahedra
  unsigned int nv = m_TetraX_Vox.rows(), nt = m_TetraVI.rows();

  // Apply transformation to the coordinates
  for(unsigned int i = 0; i < nv; i++)
    for(unsigned int j = 0; j < VDim; j++)
      m_TetraX_RAS_Warped[i][j] = m_TetraX_RAS[i][j] + disp_ras[i][j];

  // Compute the transformed tetrahedral volumes - forward pass
  for(unsigned int k = 0; k < nt; k++)
    m_TetraVol_Warped[k] = m_TetraVolumeLayer[k].Forward(m_TetraX_RAS_Warped, true);

  // Compute the difference of Jacobians metric
  double obj = 0.0;
  m_D_TetraVol_Warped.fill(0.0);
  for(unsigned int j = 0; j < m_TetraNbr.size(); j++)
    {
      int i1, i2;
      std::tie(i1, i2) = m_TetraNbr[j];
      double jac_1 = m_TetraVol_Warped[i1] / m_TetraVol[i1];
      double jac_2 = m_TetraVol_Warped[i2] / m_TetraVol[i2];
      obj += (jac_1 - jac_2) * (jac_1 - jac_2);

      // Backprop onto the volumes of the tetrahedra
      m_D_TetraVol_Warped[i1] += 2.0 * (jac_1 - jac_2) / m_TetraVol[i1];
      m_D_TetraVol_Warped[i2] -= 2.0 * (jac_1 - jac_2) / m_TetraVol[i2];
    }

  // Final objective value
  double obj_scale = weight / m_TetraNbr.size();
  obj *= obj_scale;
  m_D_TetraVol_Warped *= obj_scale;

  // Backprop onto the RAS coordinates
  grad.fill(0.0);
  for(unsigned int k = 0; k < nt; k++)
    m_TetraVolumeLayer[k].Backward(m_D_TetraVol_Warped[k], grad);

  return obj;
}


template<class TFloat, unsigned int VDim>
double TetraMeshConstraints<TFloat, VDim>
    ::ComputeObjectiveAndGradientPhi(VectorImageType *phi_vox, VectorImageType *grad, double weight)
{
  // Number of vertices and tetrahedra
  unsigned int nv = m_TetraX_Vox.rows(), nt = m_TetraVI.rows();

  // Create a fast interpolation function
  typedef typename VectorImageType::PixelType VectorImagePixel;
  typedef FastLinearInterpolator<VectorImageType, TFloat, VDim> Interpolator;
  Interpolator interp(phi_vox);
  Interpolator interp_grad(grad);

  // Interpolate the warp at each mesh vertex
  VectorImagePixel phi_i;
  Vec phi_i_vox;

  // Sample point - has to be TFloat
  TFloat x_i[VDim];

  // Apply transformation to the coordinates
  for(unsigned int i = 0; i < nv; i++)
    {
      // Interpolate warp at this location
      for(unsigned int d = 0; d < VDim; d++)
        x_i[d] = (TFloat) m_TetraX_Vox(i,d);
      interp.Interpolate(x_i, &phi_i);

      // Transform into a RAS displacement
      for(unsigned int a = 0; a < VDim; a++)
        {
          double disp = 0.0;
          for(unsigned int b = 0; b < VDim; b++)
            disp += A_vox_to_ras[a][b] * phi_i[b];
          m_TetraX_RAS_Disp[i][a] = disp;
          m_TetraX_RAS_Warped[i][a] = m_TetraX_RAS[i][a] + disp;
        }
    }

  // Compute the gradient with respect to the displacement
  double obj = this->ComputeObjectiveAndGradientDisp(m_TetraX_RAS_Disp, m_D_TetraX_RAS_Warped, weight);

  // Backprop onto the phi values
  VectorImagePixel D_phi_i_vox;
  unsigned int n_inside = 0, n_outside = 0, n_border = 0;
  for(unsigned int i = 0; i < nv; i++)
    {
      // Calculate the splat position and the splatted vector (derivative of objective
      // with respect to the voxel-space displacement)
      for(unsigned int a = 0; a < VDim; a++)
        {
          x_i[a] = (TFloat) m_TetraX_Vox(i, a);
          D_phi_i_vox[a] = 0.0;
          for(unsigned int b = 0; b < VDim; b++)
            D_phi_i_vox[a] += A_vox_to_ras[b][a] * m_D_TetraX_RAS_Warped[i][b];
        }

      // Splat this derivative onto the phi gradient
      auto status = interp_grad.Splat(x_i, &D_phi_i_vox);
      if(status == Interpolator::INSIDE)
        n_inside++;
      else if(status == Interpolator::BORDER)
        n_border++;
      else
        n_outside++;
    }

  return obj;
}

template <unsigned int VDim>
vtkSmartPointer<vtkUnstructuredGrid> create_sample_tetra_mesh()
{
  return vtkNew<vtkUnstructuredGrid>();
}

template <>
vtkSmartPointer<vtkUnstructuredGrid> create_sample_tetra_mesh<3>()
{
  // Create a regular hexahedron
  // https://kitware.github.io/vtk-examples/site/Cxx/GeometricObjects/Cell3DDemonstration/

  // Create the points
  vtkNew<vtkPoints> points;
  points->InsertNextPoint(0.2, 0.2, 0.2);
  points->InsertNextPoint(0.8, 0.2, 0.2);
  points->InsertNextPoint(0.8, 0.8, 0.2);
  points->InsertNextPoint(0.2, 0.8, 0.2);
  points->InsertNextPoint(0.2, 0.2, 0.8);
  points->InsertNextPoint(0.8, 0.2, 0.8);
  points->InsertNextPoint(0.8, 0.8, 0.8);
  points->InsertNextPoint(0.2, 0.8, 0.8);

  // Create the tetrahedra
  vtkIdType ids[5][4] = {
    { 0, 1, 3, 4 },
    { 1, 4, 5, 6 },
    { 1, 4, 6, 3 },
    { 1, 3, 6, 2 },
    { 3, 6, 7, 4 }
  };

  // Add the points and hexahedron to an unstructured grid
  vtkSmartPointer<vtkUnstructuredGrid> uGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  uGrid->SetPoints(points);
  for(unsigned int i = 0; i < 5; i++)
    uGrid->InsertNextCell(VTK_TETRA, 4, ids[i]);

  // Perform subdivision
  vtkNew<vtkSubdivideTetra> sub;
  sub->SetInputData(uGrid);
  sub->Update();
  vtkSmartPointer<vtkUnstructuredGrid> uGridSub = sub->GetOutput();

  // WriteMesh(uGridSub, "/tmp/ugrid.vtk");
  return uGridSub;
  // return uGrid;
}

template <>
vtkSmartPointer<vtkUnstructuredGrid> create_sample_tetra_mesh<2>()
{
  // Create a regular hexahedron
  // https://kitware.github.io/vtk-examples/site/Cxx/GeometricObjects/Cell3DDemonstration/

  // Create the points
  vtkNew<vtkPoints> points;
  points->InsertNextPoint(0.2, 0.2, 0.0);
  points->InsertNextPoint(0.8, 0.2, 0.0);
  points->InsertNextPoint(0.8, 0.8, 0.0);
  points->InsertNextPoint(0.2, 0.8, 0.0);

  // Create the triangles
  vtkIdType ids[5][4] = {
    { 0, 1, 3 },
    { 0, 2, 3 }
  };

  // Add the points and hexahedron to an unstructured grid
  vtkSmartPointer<vtkUnstructuredGrid> uGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  uGrid->SetPoints(points);
  for(unsigned int i = 0; i < 2; i++)
    uGrid->InsertNextCell(VTK_TRIANGLE, 3, ids[i]);

  return uGrid;
}

template<class TFloat, unsigned int VDim>
bool TetraMeshConstraints<TFloat, VDim>::TestDerivatives(std::mt19937 &rnd, ImageBaseType *refspace, vtkUnstructuredGrid *mesh)
{
  double eps = 0.001;

  // Create a tetrahedral mesh of appropriate dimensions
  vtkSmartPointer<vtkUnstructuredGrid> tetra = mesh;
  if(!mesh)
    tetra = create_sample_tetra_mesh<3>();

  // Create a dummy image
  typename VectorImageType::Pointer phi;
  if(refspace)
    {
      phi = LDDMMType::new_vimg(refspace);
      LDDMMType::vimg_add_gaussian_noise_in_place(phi, 1.0, rnd);
      LDDMMType::vimg_smooth(phi, phi, 2.0);
    }
  else
    phi = DisplacementSelfCompositionLayer<VDim, TFloat>::MakeTestDisplacement(32, 8.0, 1.0, true);

  // Generate a gradient image
  typename LDDMMType::VectorImagePointer grad = LDDMMType::new_vimg(phi, 0.0);

  // Generate a random variation of phi
  typename LDDMMType::VectorImagePointer variation = LDDMMType::new_vimg(phi, 0.0);
  LDDMMType::vimg_add_gaussian_noise_in_place(variation, 1.0, rnd);
  LDDMMType::vimg_smooth(variation, variation, 1.2);

  // What weight to use
  double weight = 4.0;

  // Initialize the mesh computation
  TetraMeshConstraints<TFloat, VDim> tmc;
  tmc.SetMesh(tetra);
  tmc.SetReferenceImage(phi);

  std::cout << tmc.A_vox_to_ras << std::endl;

  // First, let's test the mesh portion of the network. We will apply a random set of
  // displacements to the mesh coordinates and check the derivatives
  std::normal_distribution<TFloat> ndist(0., 1.);
  vnl_matrix<double> disp_x_ras(tmc.m_TetraX_RAS.rows(), VDim);
  vnl_matrix<double> grad_disp_x_ras(tmc.m_TetraX_RAS.rows(), VDim);
  vnl_matrix<double> variation_x_ras(tmc.m_TetraX_RAS.rows(), VDim);
  auto tet_x_ras_warped = tmc.m_TetraX_RAS;
  for(unsigned int i = 0; i < tmc.m_TetraX_RAS.rows(); i++)
    {
      for(unsigned int j = 0; j < VDim; j++)
        {
          disp_x_ras[i][j] = ndist(rnd) * 1.0;
          variation_x_ras[i][j] = ndist(rnd) * 1.0;
        }
    }

  // Compute the numerical derivative
  double obj_plus = tmc.ComputeObjectiveAndGradientDisp(
      disp_x_ras + variation_x_ras * eps, grad_disp_x_ras, weight);
  double obj_minus = tmc.ComputeObjectiveAndGradientDisp(
      disp_x_ras - variation_x_ras * eps, grad_disp_x_ras, weight);

  double num_deriv = (obj_plus - obj_minus) / (2.0 * eps);

  // Compute the analytical derivative
  double obj = tmc.ComputeObjectiveAndGradientDisp(disp_x_ras, grad_disp_x_ras, weight);
  double ana_deriv = dot_product(grad_disp_x_ras, variation_x_ras);

  // Compute relative difference
  double rel_diff = 2.0 * std::fabs(ana_deriv - num_deriv) / (1.0e-8 + std::fabs(ana_deriv) + std::fabs(num_deriv));

  // Compute the difference between the two derivatives
  printf("Derivatives (Mesh): ANA: %12.8g  NUM: %12.8g  RELDIF: %12.8f\n", ana_deriv, num_deriv, rel_diff);

  // Now perform the computation with respect to a displacement field
  obj = tmc.ComputeObjectiveAndGradientPhi(phi, grad, weight);

  // Report the tetrahedral volumes
  printf("Objective: %8.6f\n", obj);

  // We only want to report a subset of vertices
  int n_report = 100;
  int report_interval = std::max(1, (int) (tmc.m_TetraVol.size() / n_report));
  for(unsigned int i = 0; i < tmc.m_TetraVol.size(); i+=report_interval)
    printf("Tetra %3d Volume, fixed = %12.9f, warped = %12.9f\n", i, tmc.m_TetraVol[i], tmc.m_TetraVol_Warped[i]);

  report_interval = std::max(1, (int) (tmc.m_TetraNbr.size() / n_report));
  for(unsigned int i = 0; i < tmc.m_TetraNbr.size(); i+=report_interval)
    {
      int i1, i2;
      std::tie(i1, i2) = tmc.m_TetraNbr[i];
      double v1 = tmc.m_TetraVol[i1], v2 = tmc.m_TetraVol[i2];
      double w1 = tmc.m_TetraVol_Warped[i1], w2 = tmc.m_TetraVol_Warped[i2];
      double jac1 = w1 / v1, jac2 = w2 / v2;
      printf("Pair %d, %d  Jac = %12.9f / %12.9f  SD = %12.9f\n", i1, i2, jac1, jac2, (jac1 - jac2) * (jac1 - jac2));
    }

  // Compute the analytic derivative with respect to variation
  typename LDDMMType::ImagePointer idot = LDDMMType::new_img(phi, 0.0);
  LDDMMType::vimg_euclidean_inner_product(idot, grad, variation);
  ana_deriv = LDDMMType::img_voxel_sum(idot);

  // Compute the numeric derivative with respect to variation
  LDDMMType::vimg_add_scaled_in_place(phi, variation, eps);
  obj_plus = tmc.ComputeObjectiveAndGradientPhi(phi, grad, weight);
  LDDMMType::vimg_add_scaled_in_place(phi, variation, -2.0 * eps);
  obj_minus = tmc.ComputeObjectiveAndGradientPhi(phi, grad, weight);
  num_deriv = (obj_plus - obj_minus) / (2.0 * eps);

  // Compute relative difference
  rel_diff = 2.0 * std::fabs(ana_deriv - num_deriv) / (1.0e-8 + std::fabs(ana_deriv) + std::fabs(num_deriv));

  // Compute the difference between the two derivatives
  printf("Derivatives (Warp): ANA: %12.8g  NUM: %12.8g  RELDIF: %12.8f\n", ana_deriv, num_deriv, rel_diff);

  return rel_diff < 1.0e-4;
}


template class TetraMeshConstraints<float, 2>;
template class TetraMeshConstraints<float, 3>;
template class TetraMeshConstraints<float, 4>;
template class TetraMeshConstraints<double, 2>;
template class TetraMeshConstraints<double, 3>;
template class TetraMeshConstraints<double, 4>;
