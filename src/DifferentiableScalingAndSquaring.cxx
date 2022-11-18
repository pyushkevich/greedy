#include "DifferentiableScalingAndSquaring.h"
#include "FastLinearInterpolator.h"
#include "ImageRegionConstIteratorWithIndexOverride.h"
#include "itkMultiThreaderBase.h"
#include "itkTimeProbe.h"
#include "lddmm_common.h"
#include <vnl/vnl_random.h>
#include <functional>


template<unsigned int VDim, typename TReal>
void DisplacementSelfCompositionLayer<VDim, TReal>::Forward(VectorImageType *u, VectorImageType *v)
{
  // Create an iterator over the deformation field
  typedef itk::ImageLinearIteratorWithIndex<VectorImageType> IterBase;
  typedef IteratorExtender<IterBase> IterType;
  typedef FastLinearInterpolator<VectorImageType, TReal, VDim> InterpType;

  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
  mt->ParallelizeImageRegion<VDim>(
        v->GetBufferedRegion(),
        [u,v](const itk::ImageRegion<VDim> &region)
  {
    // Interpolator for the image u
    InterpType fi(u);

    // Loop over the lines in the image
    int line_len = region.GetSize(0);
    for(IterType it(v, region); !it.IsAtEnd(); it.NextLine())
      {
      // Get the pointer to the u line and the output line
      const auto *u_line = it.GetPixelPointer(u);
      auto *v_line = it.GetPixelPointer(v);

      // Voxel index
      auto idx = it.GetIndex();

      // The current sample position
      itk::ContinuousIndex<TReal, VDim> cix;

      // Loop over the line
      for(int i = 0; i < line_len; i++, u_line++, v_line++)
        {
        for(unsigned int j = 0; j < VDim; j++)
          cix[j] = idx[j] + (*u_line)[j];
        idx[0]++;

        // Perform the interpolation
        auto status = fi.Interpolate(cix.GetDataPointer(), v_line);
        if(status == InterpType::OUTSIDE)
          v_line->Fill(0.0);

        // Add u to v
        (*v_line) += (*u_line);
        }
      }
  }, nullptr);
}

template<unsigned int VDim, typename TReal>
void DisplacementSelfCompositionLayer<VDim, TReal>::ForwardSingleThreaded(VectorImageType *u, VectorImageType *v)
{
  // Create an iterator over the deformation field
  typedef itk::ImageLinearIteratorWithIndex<VectorImageType> IterBase;
  typedef IteratorExtender<IterBase> IterType;
  typedef FastLinearInterpolator<VectorImageType, TReal, VDim> InterpType;

  InterpType fi(u);

  // Loop over the lines in the image
  int line_len = v->GetBufferedRegion().GetSize(0);
  itk::ContinuousIndex<TReal, VDim> cix;
  for(IterType it(v, v->GetBufferedRegion()); !it.IsAtEnd(); it.NextLine())
    {
    // Get the pointer to the u line and the output line
    const auto *u_line = it.GetPixelPointer(u);
    auto *v_line = it.GetPixelPointer(v);

    // Voxel index
    auto idx = it.GetIndex();

    // Loop over the line
    for(int i = 0; i < line_len; i++, u_line++, v_line++)
      {
      for(unsigned int j = 0; j < VDim; j++)
        cix[j] = idx[j] + (*u_line)[j];
      idx[0]++;

      // Perform the interpolation
      auto status = fi.Interpolate(cix.GetDataPointer(), v_line);
      if(status == InterpType::OUTSIDE)
        v_line->Fill(0.0);

      // Add u to v
      (*v_line) += (*u_line);
      }
    }
}


template<unsigned int VDim, typename TReal>
void DisplacementSelfCompositionLayer<VDim, TReal>
::Backward(VectorImageType *u, VectorImageType *Dv_f, VectorImageType *Du_f)
{
  // Create an iterator over the deformation field
  typedef itk::ImageLinearIteratorWithIndex<VectorImageType> IterBase;
  typedef IteratorExtender<IterBase> IterType;
  typedef typename VectorImageType::PixelType VectorType;
  typedef FastLinearInterpolator<VectorImageType, TReal, VDim> InterpType;

  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();

  // We need to find the biggest offset - this is so that in the second pass
  // we don't have to scan the entire image
  VectorType global_u_min(0.0), global_u_max(0.0);
  std::mutex u_mutex;

  // First pass, during which we compute the component of the gradient that involves
  // the u itself and set up the variables for the splatting pass.
  mt->ParallelizeImageRegion<VDim>(
        Du_f->GetBufferedRegion(),
        [u,Dv_f,Du_f,&u_mutex,&global_u_min,&global_u_max](const itk::ImageRegion<VDim> &region)
  {
    InterpType fi(u);

    // Storage for the sampling coordiante
    itk::ContinuousIndex<TReal, VDim> cix;

    // Storage for the gradient of the interpolated function
    VectorType v, *Dx_v = new VectorType[VDim];

    // Maximum offset
    VectorType u_min(0.0), u_max(0.0);

    // Loop over the lines in the image
    int line_len = region.GetSize(0);
    for(IterType it(Du_f, region); !it.IsAtEnd(); it.NextLine())
      {
      // Get the pointer to the u line and the output line
      const auto *u_line = it.GetPixelPointer(u);
      const auto *Dv_f_line = it.GetPixelPointer(Dv_f);
      auto *Du_f_line = it.GetPixelPointer(Du_f);

      // const auto *offset_line = it.GetPixelPointer(this->m_OffsetImage.GetPointer());
      // const auto *remainder_line = it.GetPixelPointer(this->m_RemainderImage.GetPointer());

      // Voxel index
      auto idx = it.GetIndex();

      // Loop over the line
      for(int i = 0; i < line_len; i++, u_line++, Dv_f_line++, Du_f_line++)
        {
        const auto &uvec = *u_line;
        for(unsigned int j = 0; j < VDim; j++)
          {
          cix[j] = idx[j] + uvec[j];
          if(u_min[j] > uvec[j]) u_min[j] = uvec[j];
          if(u_max[j] < uvec[j]) u_max[j] = uvec[j];
          }
        idx[0]++;

        // The expression is v = u + interp(u, x+u)
        // The backprop is Du_f = Dv_f + D1_interp(u, x+u) Du_f + D2_interp(u, x+u) Du_f

        // First term
        (*Du_f_line) += (*Dv_f_line);

        // Second term
        auto status = fi.InterpolateWithGradient(cix.GetDataPointer(), &v, &Dx_v);
        if(status != InterpType::OUTSIDE)
          {
          for(unsigned int a = 0; a < VDim; a++)
            for(unsigned int b = 0; b < VDim; b++)
              (*Du_f_line)[a] += Dx_v[a][b] * (*Dv_f_line)[b];
          }
        }
      }

    // Cleanup
    delete[] Dx_v;

    // Use mutex to update the u range variables
    std::lock_guard<std::mutex> guard(u_mutex);
    for(unsigned int j = 0; j < VDim; j++)
      {
      if(global_u_min[j] > u_min[j]) global_u_min[j] = u_min[j];
      if(global_u_max[j] < u_max[j]) global_u_max[j] = u_max[j];
      }

  }, nullptr);

  // printf("Got here\n");
  // std::cout << global_u_min << std::endl;
  // std::cout << global_u_max << std::endl;

  // Second pass during which we splat
  mt->ParallelizeImageRegion<VDim>(
        Du_f->GetBufferedRegion(),
        [u,Dv_f,Du_f,&global_u_min,&global_u_max](const itk::ImageRegion<VDim> &region)
  {
    // Create an interpolator limited to the target region
    InterpType fi_splat(Du_f, region);

    // Storage for the sampling coordiante
    itk::ContinuousIndex<TReal, VDim> cix;

    // Determine the region that we must scan in order to find all the pixels
    // that splat to the current threaded region
    itk::ImageRegion<VDim> search_region = region;
    for(unsigned int i = 0; i < VDim; i++)
      {
      int o1 = std::ceil(-global_u_min[i]), o2 = std::ceil(global_u_max[i]);
      // int offset = std::ceil(std::max(-global_u_min[i], global_u_max[i]));
      search_region.SetIndex(i, region.GetIndex(i) - o2);
      search_region.SetSize(i, region.GetSize(i) + o1 + o2);
      }
    search_region.Crop(Du_f->GetBufferedRegion());

    // Loop over the lines in the image - this time we are iterating over the whole image
    // and not just the region belonging to this thread
    int line_len = search_region.GetSize(0);
    for(IterType it(Du_f, search_region); !it.IsAtEnd(); it.NextLine())
      {
      // Get the pointer to the u line and the output line
      const auto *u_line = it.GetPixelPointer(u);
      const auto *Dv_f_line = it.GetPixelPointer(Dv_f);

      // Voxel index
      auto idx = it.GetIndex();

      // Loop over the line
      for(int i = 0; i < line_len; i++, u_line++, Dv_f_line++)
        {
        // Get the current coordinate
        for(unsigned int j = 0; j < VDim; j++)
          cix[j] = idx[j] + (*u_line)[j];
        idx[0]++;

        // Check the coordinate against bounds of the image region
        fi_splat.Splat(cix.GetDataPointer(), Dv_f_line);
        }
      }
  }, nullptr);
}

template<unsigned int VDim, typename TReal>
void DisplacementSelfCompositionLayer<VDim, TReal>
::BackwardSingleThreaded(VectorImageType *u, VectorImageType *Dv_f, VectorImageType *Du_f)
{
  // Create an iterator over the deformation field
  typedef itk::ImageLinearIteratorWithIndex<VectorImageType> IterBase;
  typedef IteratorExtender<IterBase> IterType;
  typedef typename VectorImageType::PixelType VectorType;
  typedef FastLinearInterpolator<VectorImageType, TReal, VDim> InterpType;

  InterpType fi(u);
  InterpType fi_splat(Du_f);

  // Storage for the gradient
  VectorType v, *Dx_v = new VectorType[VDim];

  // Loop over the lines in the image
  int line_len = Du_f->GetBufferedRegion().GetSize(0);
  for(IterType it(Du_f, Du_f->GetBufferedRegion()); !it.IsAtEnd(); it.NextLine())
    {
    // Get the pointer to the u line and the output line
    const auto *u_line = it.GetPixelPointer(u);
    const auto *Dv_f_line = it.GetPixelPointer(Dv_f);
    auto *Du_f_line = it.GetPixelPointer(Du_f);

    // Voxel index
    auto idx = it.GetIndex();

    // The current sample position
    itk::ContinuousIndex<TReal, VDim> cix;

    // Loop over the line
    for(int i = 0; i < line_len; i++, u_line++, Dv_f_line++, Du_f_line++)
      {
      for(unsigned int j = 0; j < VDim; j++)
        cix[j] = idx[j] + (*u_line)[j];
      idx[0]++;

      // The expression is v = u + interp(u, x+u)
      // The backprop is Du_f = Dv_f + D1_interp(u, x+u) Du_f + D2_interp(u, x+u) Du_f

      // First term
      (*Du_f_line) += (*Dv_f_line);

      // Second term
      auto status = fi.InterpolateWithGradient(cix.GetDataPointer(), &v, &Dx_v);
      if(status != InterpType::OUTSIDE)
        {
        // for(unsigned int d = 0; d < VDim; d++)
        //  (*Du_f_line) += Dx_v[d] * (*Dv_f_line)[d];
        for(unsigned int a = 0; a < VDim; a++)
          for(unsigned int b = 0; b < VDim; b++)
            (*Du_f_line)[a] += Dx_v[a][b] * (*Dv_f_line)[b];
        }

      // Third term
      fi_splat.Splat(cix.GetDataPointer(), Dv_f_line);
      }
    }

  // Cleanup
  delete [] Dx_v;
}

template<unsigned int VDim, typename TReal>
typename DisplacementSelfCompositionLayer<VDim, TReal>::VectorImageType::Pointer
DisplacementSelfCompositionLayer<VDim, TReal>::MakeTestDisplacement(
    int size, TReal scale, TReal sigma, bool orient_ras)
{
  // Create a dummy image
  typename VectorImageType::Pointer phi = VectorImageType::New();
  typename VectorImageType::SizeType sz_phi;
  typename VectorImageType::RegionType region;
  double origin_phi[VDim], spacing_phi[VDim];
  for(unsigned int d = 0; d < VDim; d++)
    {
    sz_phi[d] = size;
    spacing_phi[d] = 1.0 / sz_phi[d];
    origin_phi[d] = 0.5 * spacing_phi[d];
    }

  region.SetSize(sz_phi);
  phi->SetOrigin(origin_phi);
  phi->SetSpacing(spacing_phi);
  phi->SetRegions(region);

  if(orient_ras)
    {
    typename VectorImageType::DirectionType dir_phi;
    dir_phi.Fill(0.0);
    for(unsigned int d = 0; d < VDim; d++)
      dir_phi(d,d) = d < 2 ? -1.0 : 1.0;
    phi->SetDirection(dir_phi);
    }

  phi->Allocate();

  // Fill image with random noise
  vnl_random randy;
  for(itk::ImageRegionIteratorWithIndex<VectorImageType> it(phi, region); !it.IsAtEnd(); ++it)
    for(unsigned int d = 0; d < VDim; d++)
      it.Value()[d] = randy.normal() * scale;
  LDDMMType::vimg_smooth(phi, phi, sigma);

  return phi;
}


template<unsigned int VDim, typename TReal>
bool DisplacementSelfCompositionLayer<VDim, TReal>::TestDerivatives()
{
  // Create the self-composition image
  typename VectorImageType::Pointer phi = MakeTestDisplacement(96, 8.0, 1.0);
  typename VectorImageType::Pointer phi_comp_phi_1 = LDDMMType::new_vimg(phi);
  typename VectorImageType::Pointer phi_comp_phi_2 = LDDMMType::new_vimg(phi);
  typename VectorImageType::Pointer phi_comp_phi_3 = LDDMMType::new_vimg(phi);

  // Compute the composition the normal way
  LDDMMType::interp_vimg(phi, phi, 1.0, phi_comp_phi_1);
  LDDMMType::vimg_add_in_place(phi_comp_phi_1, phi);

  // Compute the composition the way this class does
  DisplacementSelfCompositionLayer<VDim, TReal> self;
  itk::TimeProbe tp_f_mt, tp_f_st;
  tp_f_mt.Start();
  self.Forward(phi, phi_comp_phi_2);
  tp_f_mt.Stop();

  tp_f_st.Start();
  self.ForwardSingleThreaded(phi, phi_comp_phi_3);
  tp_f_st.Stop();

  printf("Forward run time ST: %f, MT: %f\n", tp_f_st.GetTotal(), tp_f_mt.GetTotal());

  // Compare the two results
  LDDMMType::vimg_subtract_in_place(phi_comp_phi_1, phi_comp_phi_2);
  double err1 = LDDMMType::vimg_euclidean_norm_sq(phi_comp_phi_1);
  printf("Error Forward vs LDDMMType::interp_vimg: %12.8f\n", err1);

  // Compare the two results
  LDDMMType::vimg_subtract_in_place(phi_comp_phi_2, phi_comp_phi_3);
  double err2 = LDDMMType::vimg_euclidean_norm_sq(phi_comp_phi_2);
  printf("Error Forward vs ForwardSingleThreaded: %12.8f\n", err2);

  // Let's define the objective function as just the norm of the vector field
  double nvox = phi_comp_phi_3->GetBufferedRegion().GetNumberOfPixels();
  double obj = LDDMMType::vimg_euclidean_norm_sq(phi_comp_phi_3) / nvox;

  // Then the derivative of the objective with respect to phi_comp_phi_2 is just 2 * phi_comp_phi_2 / nvox;
  typename VectorImageType::Pointer D_v_obj = LDDMMType::new_vimg(phi);
  LDDMMType::vimg_copy(phi_comp_phi_3, D_v_obj);
  LDDMMType::vimg_scale_in_place(D_v_obj, 2.0 / nvox);

  // Backpropagate this to the derivative with respect to u
  typename VectorImageType::Pointer D_u_obj = LDDMMType::new_vimg(phi);
  typename VectorImageType::Pointer D_u_obj_st = LDDMMType::new_vimg(phi);

  itk::TimeProbe tp_mt, tp_st;
  tp_mt.Start();
  self.Backward(phi, D_v_obj, D_u_obj);
  tp_mt.Stop();

  tp_st.Start();
  self.BackwardSingleThreaded(phi, D_v_obj, D_u_obj_st);
  tp_st.Stop();

  printf("Run time ST: %f, MT: %f\n", tp_st.GetTotal(), tp_mt.GetTotal());

  // Compare the two results
  LDDMMType::vimg_subtract_in_place(D_u_obj_st, D_u_obj);
  double err3 = LDDMMType::vimg_euclidean_norm_sq(D_u_obj_st);
  printf("Error Backward vs BackwardSingleThreaded: %12.8f\n", err3);

  // Generate a random variation of phi
  typename LDDMMType::VectorImagePointer variation = MakeTestDisplacement(96, 1.0, 0.2);

  // Compute the analytic derivative with respect to variation
  typename LDDMMType::ImagePointer idot = LDDMMType::new_img(phi, 0.0);
  LDDMMType::vimg_euclidean_inner_product(idot, D_u_obj, variation);
  double ana_deriv = LDDMMType::img_voxel_sum(idot);

  // Compute the numeric derivative with respect to variation
  double eps = 0.001;
  typename LDDMMType::VectorImagePointer work = LDDMMType::new_vimg(phi, 0.0);
  LDDMMType::vimg_add_scaled_in_place(phi, variation, eps);
  self.Forward(phi, work);
  double obj_plus = LDDMMType::vimg_euclidean_norm_sq(work) / nvox;
  LDDMMType::vimg_add_scaled_in_place(phi, variation, -2.0 * eps);
  self.Forward(phi, work);
  double obj_minus = LDDMMType::vimg_euclidean_norm_sq(work) / nvox;
  double num_deriv = (obj_plus - obj_minus) / (2.0 * eps);

  // Compute relative difference
  double rel_diff = 2.0 * std::fabs(ana_deriv - num_deriv) / std::fabs(ana_deriv + num_deriv);

  // Compute the difference between the two derivatives
  printf("Derivatives: ANA: %12.8g  NUM: %12.8g  RELDIF: %12.8f\n", ana_deriv, num_deriv, rel_diff);

  return rel_diff < 1.0e-4;
}

template class DisplacementSelfCompositionLayer<2, float>;
template class DisplacementSelfCompositionLayer<3, float>;
template class DisplacementSelfCompositionLayer<4, float>;
template class DisplacementSelfCompositionLayer<2, double>;
template class DisplacementSelfCompositionLayer<3, double>;
template class DisplacementSelfCompositionLayer<4, double>;

template<unsigned int VDim, typename TReal>
ScalingAndSquaringLayer<VDim, TReal>
::ScalingAndSquaringLayer(VectorImageType *u, unsigned int n_steps)
{
  this->m_Steps = n_steps;
  this->m_WorkImage.resize(this->m_Steps, nullptr);
  for(unsigned int i = 0; i < this->m_Steps; i++)
    this->m_WorkImage[i] = LDDMMType::new_vimg(u);
}

template<unsigned int VDim, typename TReal>
void ScalingAndSquaringLayer<VDim, TReal>::Forward(VectorImageType *u, VectorImageType *v)
{
  for(unsigned int i = 0; i < m_Steps; i++)
    m_CompositionLayer.Forward(i == 0 ? u : m_WorkImage[i-1].GetPointer(),
                               i == m_Steps - 1 ? v : m_WorkImage[i].GetPointer());
}

template<unsigned int VDim, typename TReal>
void ScalingAndSquaringLayer<VDim, TReal>
::Backward(VectorImageType *u, VectorImageType *Dv_f, VectorImageType *Du_f)
{

  for(int i = m_Steps-1; i >= 0; i--)
    {
    if(i > 0)
      m_WorkImage[i]->FillBuffer(Vec(0.0));
    m_CompositionLayer.Backward(i == 0 ? u : m_WorkImage[i-1].GetPointer(),
                                i == (int) (m_Steps - 1) ? Dv_f : m_WorkImage[i+1].GetPointer(),
                                i == 0 ? Du_f : m_WorkImage[i].GetPointer());
    }
}

template<unsigned int VDim, typename TReal>
bool ScalingAndSquaringLayer<VDim, TReal>::TestDerivatives()
{
  // Create the stationary velocity field
  typename VectorImageType::Pointer u = CompositionLayer::MakeTestDisplacement(96, 8.0, 1.0);
  typename VectorImageType::Pointer v_reference = LDDMMType::new_vimg(u);
  typename VectorImageType::Pointer v_test = LDDMMType::new_vimg(u);
  typename VectorImageType::Pointer work = LDDMMType::new_vimg(u);

  // Create the test object
  ScalingAndSquaringLayer<VDim, TReal> self(u);

  // Perform scaling and squaring the conventional way
  itk::TimeProbe tp_f_reference, tp_f_test;
  tp_f_reference.Start();
  LDDMMType::vimg_exp(u, v_reference, work, 6, 1.0);
  tp_f_reference.Stop();

  // Perform scaling and squaring using the layer
  tp_f_test.Start();
  self.Forward(u, v_test);
  tp_f_test.Stop();

  // Print forward time statistics
  printf("Forward run time reference: %f, test: %f\n", tp_f_reference.GetTotal(), tp_f_test.GetTotal());

  // Compare the two results
  LDDMMType::vimg_subtract_in_place(v_reference, v_test);
  double err1 = LDDMMType::vimg_euclidean_norm_sq(v_reference);
  printf("Error Test vs Reference: %12.8f\n", err1);

  // Let's define the objective function as just the norm of the vector field
  double nvox = u->GetBufferedRegion().GetNumberOfPixels();
  double obj = LDDMMType::vimg_euclidean_norm_sq(v_test) / nvox;

  // Then the derivative of the objective with respect to phi_comp_phi_2 is just 2 * phi_comp_phi_2 / nvox;
  typename VectorImageType::Pointer D_v_obj = LDDMMType::new_vimg(u);
  LDDMMType::vimg_copy(v_test, D_v_obj);
  LDDMMType::vimg_scale_in_place(D_v_obj, 2.0 / nvox);

  // Backpropagate this to the derivative with respect to u
  typename VectorImageType::Pointer D_u_obj = LDDMMType::new_vimg(u);

  itk::TimeProbe tp_b;
  tp_b.Start();
  self.Backward(u, D_v_obj, D_u_obj);
  tp_b.Stop();

  printf("Run time backprop: %f\n", tp_b.GetTotal());

  // Generate a random variation of phi
  typename LDDMMType::VectorImagePointer variation = CompositionLayer::MakeTestDisplacement(96, 1.0, 0.2);

  // Compute the analytic derivative with respect to variation
  typename LDDMMType::ImagePointer idot = LDDMMType::new_img(u, 0.0);
  LDDMMType::vimg_euclidean_inner_product(idot, D_u_obj, variation);
  double ana_deriv = LDDMMType::img_voxel_sum(idot);

  // Compute the numeric derivative with respect to variation
  double eps = 0.001;
  LDDMMType::vimg_add_scaled_in_place(u, variation, eps);
  self.Forward(u, work);
  double obj_plus = LDDMMType::vimg_euclidean_norm_sq(work) / nvox;
  LDDMMType::vimg_add_scaled_in_place(u, variation, -2.0 * eps);
  self.Forward(u, work);
  double obj_minus = LDDMMType::vimg_euclidean_norm_sq(work) / nvox;
  double num_deriv = (obj_plus - obj_minus) / (2.0 * eps);

  // Compute relative difference
  double rel_diff = 2.0 * std::fabs(ana_deriv - num_deriv) / std::fabs(ana_deriv + num_deriv);

  // Compute the difference between the two derivatives
  printf("Derivatives: ANA: %12.8g  NUM: %12.8g  RELDIF: %12.8f\n", ana_deriv, num_deriv, rel_diff);

  return rel_diff < 1.0e-4;
}

template class ScalingAndSquaringLayer<2, float>;
template class ScalingAndSquaringLayer<3, float>;
template class ScalingAndSquaringLayer<4, float>;
template class ScalingAndSquaringLayer<2, double>;
template class ScalingAndSquaringLayer<3, double>;
template class ScalingAndSquaringLayer<4, double>;


template<unsigned int VDim, typename TReal>
double DisplacementFieldSmoothnessLoss<VDim, TReal>
::ComputeLossAndGradient(VectorImageType *u, VectorImageType *grad, TReal grad_scale)
{
  // Create an iterator over the deformation field
  typedef itk::ImageLinearIteratorWithIndex<VectorImageType> IterBase;
  typedef IteratorExtenderWithOffset<IterBase> IterType;

  // Scaling factor for the derivative
  double scale = grad_scale * 2.0 / u->GetBufferedRegion().GetNumberOfPixels();

  // Loop over the image dimension
  double obj = 0.0;
  for(unsigned int d = 0; d < VDim; d++)
    {
    std::mutex mutex;
    itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
    mt->ParallelizeImageRegionRestrictDirection<VDim>(
          d, u->GetBufferedRegion(),
          [u, d, &mutex, &obj, scale, &grad](const itk::ImageRegion<VDim> &region)
      {
      IterType it(u, region);
      it.SetDirection(d);
      int line_len = region.GetSize(d);
      int jump = it.GetOffset(d);
      double thread_obj = 0.0;

      for(it.GoToBegin(); !it.IsAtEnd(); it.NextLine())
        {
        const Vec *p = it.GetPosition();
        Vec *p_grad = it.GetPixelPointer(grad);
        for(unsigned int i = 0; i < line_len - 1; i++, p+=jump, p_grad+=jump)
          {
          const Vec &a = *p, &b = *(p + jump);
          Vec &d_a = *p_grad, &d_b = *(p_grad + jump);
          for(unsigned int k = 0; k < VDim; k++)
            {
            TReal del = (b[k] - a[k]);
            TReal del_scaled = del * scale;
            d_a[k] -= del_scaled;
            d_b[k] += del_scaled;
            thread_obj += del * del;
            }
          }
        }

      std::lock_guard<std::mutex> guard(mutex);
      obj += thread_obj;
      }, nullptr);
    }

  return obj / u->GetBufferedRegion().GetNumberOfPixels();
}

template<unsigned int VDim, typename TReal>
bool DisplacementFieldSmoothnessLoss<VDim, TReal>::TestDerivatives()
{
  // Create the stationary velocity field
  typedef DisplacementSelfCompositionLayer<VDim, TReal> CompositionLayer;
  typename VectorImageType::Pointer u = CompositionLayer::MakeTestDisplacement(96, 8.0, 1.0);
  typename VectorImageType::Pointer D_u_obj = LDDMMType::new_vimg(u);

  // Create the test object
  DisplacementFieldSmoothnessLoss<VDim, TReal> self;

  // Compute the objective and gradient in one go
  itk::TimeProbe tp_f;
  tp_f.Start();
  double obj = self.ComputeLossAndGradient(u, D_u_obj);
  tp_f.Stop();

  // Print forward time statistics
  printf("Complete run time: %f\n", tp_f.GetTotal());
  printf("Loss value: %f\n", obj);

  // Generate a random variation of phi
  typename LDDMMType::VectorImagePointer variation = CompositionLayer::MakeTestDisplacement(96, 1.0, 0.2);

  // Compute the analytic derivative with respect to variation
  typename LDDMMType::ImagePointer idot = LDDMMType::new_img(u, 0.0);
  LDDMMType::vimg_euclidean_inner_product(idot, D_u_obj, variation);
  double ana_deriv = LDDMMType::img_voxel_sum(idot);

  // Compute the numeric derivative with respect to variation
  double eps = 0.001;
  LDDMMType::vimg_add_scaled_in_place(u, variation, eps);
  double obj_plus = self.ComputeLossAndGradient(u, D_u_obj);
  LDDMMType::vimg_add_scaled_in_place(u, variation, -2.0 * eps);
  double obj_minus = self.ComputeLossAndGradient(u, D_u_obj);
  double num_deriv = (obj_plus - obj_minus) / (2.0 * eps);

  // Compute relative difference
  double rel_diff = 2.0 * std::fabs(ana_deriv - num_deriv) / std::fabs(ana_deriv + num_deriv);

  // Compute the difference between the two derivatives
  printf("Derivatives: ANA: %12.8g  NUM: %12.8g  RELDIF: %12.8f\n", ana_deriv, num_deriv, rel_diff);

  return rel_diff < 1.0e-4;
}

template class DisplacementFieldSmoothnessLoss<2, float>;
template class DisplacementFieldSmoothnessLoss<3, float>;
template class DisplacementFieldSmoothnessLoss<4, float>;
template class DisplacementFieldSmoothnessLoss<2, double>;
template class DisplacementFieldSmoothnessLoss<3, double>;
template class DisplacementFieldSmoothnessLoss<4, double>;

template <typename TAtomic>
class AdamStepHelper
{
  static void pixel_update(const TAtomic &grad_i, TAtomic &m_k_i, TAtomic &v_k_i, TAtomic &theta_i,
                           double alpha, double beta_1, double beta_2, double beta_1_t, double beta_2_t, double eps)
    {
    m_k_i = m_k_i * beta_1 + grad_i * (1.0 - beta_1);
    v_k_i = v_k_i * beta_2 + (grad_i * grad_i) * (1.0 - beta_2);
    TAtomic denom = eps + sqrt(v_k_i / (1.0 - beta_2_t));
    theta_i -= m_k_i * alpha / (denom * (1.0 - beta_1_t));
    }
};

template <unsigned int VDim, typename TReal>
class AdamStepHelper<itk::CovariantVector<TReal, VDim> >
{
public:
  typedef itk::CovariantVector<TReal, VDim> TAtomic;
  static void pixel_update(const TAtomic &grad_i, TAtomic &m_k_i, TAtomic &v_k_i, TAtomic &theta_i,
                           double alpha, double beta_1, double beta_2, double beta_1_t, double beta_2_t, double eps)
    {
    for(unsigned int d = 0; d < VDim; d++)
      {
      m_k_i[d] = m_k_i[d] * beta_1 + grad_i[d] * (1.0 - beta_1);
      v_k_i[d] = v_k_i[d] * beta_2 + (grad_i[d] * grad_i[d]) * (1.0 - beta_2);
      TReal denom = eps + sqrt(v_k_i[d] / (1.0 - beta_2_t));
      theta_i[d] -= m_k_i[d] * alpha / (denom * (1.0 - beta_1_t));
      }
    }
};

template<typename TImage>
void AdamStep<TImage>::Compute(int iter, const TImage *gradient, TImage *m_k, TImage *v_k, TImage *theta)
{
  // Create an iterator over the deformation field
  typedef itk::ImageLinearIteratorWithIndex<TImage> IterBase;
  typedef IteratorExtender<IterBase> IterType;
  typedef typename TImage::PixelType PixelType;
  typedef typename TImage::RegionType RegionType;
  typedef AdamStepHelper<PixelType> Helper;

  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
  mt->ParallelizeImageRegion<TImage::ImageDimension>(
        theta->GetBufferedRegion(),
        [gradient, m_k, v_k, theta, this, iter](const RegionType &region)
    {
    double beta_1_t = std::pow(this->beta_1, iter+1);
    double beta_2_t = std::pow(this->beta_2, iter+1);
    unsigned int line_len = region.GetSize(0);

    for(IterType it(theta, theta->GetBufferedRegion()); !it.IsAtEnd(); it.NextLine())
      {
      const auto *grad_line = it.GetPixelPointer(gradient);
      auto *m_k_line = it.GetPixelPointer(m_k);
      auto *v_k_line = it.GetPixelPointer(v_k);
      auto *theta_line = it.GetPixelPointer(theta);
      PixelType denom;

      for(unsigned int i = 0; i < line_len; i++, grad_line++, m_k_line++, v_k_line++, theta_line++)
        Helper::pixel_update(*grad_line, *m_k_line, *v_k_line, *theta_line,
                             alpha, beta_1, beta_2, beta_1_t, beta_2_t, eps);
      }
    }, nullptr);
}

template class AdamStep< typename LDDMMData<double, 2>::VectorImageType >;
template class AdamStep< typename LDDMMData<double, 3>::VectorImageType >;
template class AdamStep< typename LDDMMData<double, 4>::VectorImageType >;
template class AdamStep< typename LDDMMData<float, 2>::VectorImageType >;
template class AdamStep< typename LDDMMData<float, 3>::VectorImageType >;
template class AdamStep< typename LDDMMData<float, 4>::VectorImageType >;
