#include "PropagationAPI.h"
#include "PropagationTools.h"
#include "GreedyAPI.h"
#include "PropagationData.h"
#include "PropagationIO.h"
#include "GreedyMeshIO.h"

#include <algorithm>
#include <string>

#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkJoinSeriesImageFilter.h>
#include <itkAddImageFilter.h>
#include <vtkPolyData.h>

using namespace propagation;

template<typename TReal>
PropagationAPI<TReal>
::PropagationAPI(const std::shared_ptr<PropagationInput<TReal>> input)
{
  m_PParam = input->m_PropagationParam;
  m_GParam = input->m_GreedyParam;
  m_Data = input->m_Data;
  m_StdOut = std::make_shared<PropagationStdOut>(m_PParam.verbosity);
  ValidateInputData();
}

template<typename TReal>
void
PropagationAPI<TReal>
::ValidateInputData()
{
  if (!m_Data->img4d)
    throw GreedyException("Reference 4D Image Input not found!");

  if (!m_Data->seg_ref)
    throw GreedyException("Reference segmentation Image not found!");

  const size_t nt  = m_Data->GetNumberOfTimePoints();

  // Validate inputs
  if (m_PParam.refTP < 0 || m_PParam.refTP > nt)
    throw GreedyException("Reference tp %d cannot be greater than total number of tps %d",
                          m_PParam.refTP, nt);

  for (size_t tp : m_PParam.targetTPs)
    if (tp > nt) throw GreedyException("Target tp %d cannot be greater than total number tps %d",tp, nt);
}

template<typename TReal>
PropagationAPI<TReal>
::~PropagationAPI()
{
}

template<typename TReal>
int
PropagationAPI<TReal>
::Run()
{
	PropagationData<TReal> pData;

  if (m_PParam.debug)
		{
    m_StdOut->printf("-- [Propagation] Debug Mode is ON \n");
    m_StdOut->printf("-- [Propagation] Debug Output Dir: %s \n", m_PParam.debug_dir.c_str());
		}

  PrepareTimePointData(); // Prepare initial timepoint data for propagation
  CreateTimePointLists();

  m_StdOut->printf("-- [Propagation] forward list: ");
  for (auto tp : m_ForwardTPs)
    m_StdOut->printf(" %d", tp);
  m_StdOut->printf("\n");

  m_StdOut->printf("-- [Propagation] backward list: ");
  for (auto tp : m_BackwardTPs)
    m_StdOut->printf(" %d", tp);
  m_StdOut->printf("\n");

	// Run forward propagation
  if (m_ForwardTPs.size() > 1)
    RunUnidirectionalPropagation(m_ForwardTPs);

	// Run backward propagation
  if (m_BackwardTPs.size() > 1)
    RunUnidirectionalPropagation(m_BackwardTPs);

  // Write out a 4D Segmentation
  Generate4DSegmentation();

  m_StdOut->printf("Run Completed! \n");
  return EXIT_SUCCESS;
}

template<typename TReal>
void
PropagationAPI<TReal>
::ValidateInputOrientation()
{
  typename TImage3D::DirectionType img_direction, seg_direction;
  img_direction = m_Data->tp_data[m_PParam.refTP].img->GetDirection();
  seg_direction = m_Data->seg_ref->GetDirection();

  if (img_direction != seg_direction)
    {
    std::cerr << "Image Direction: " << std::endl << img_direction << std::endl;
    std::cerr << "Segmentation Direction: " << std::endl << seg_direction << std::endl;
    std::string fn_seg = m_PParam.use4DSegInput ? m_PParam.fn_seg4d : m_PParam.fn_seg3d;
    throw GreedyException("Image and Segmentation orientations do not match. Segmentation file %s\n",
                          fn_seg.c_str());
    }
}

template<typename TReal>
void
PropagationAPI<TReal>
::CreateReferenceMask()
{
  // Threshold, Dilate and Resample
  auto thr_tail = PTools::template ThresholdImage<TLabelImage3D, TLabelImage3D>(m_Data->seg_ref, 1, SHRT_MAX, 1, 0);
  auto dlt_tail = PTools::template DilateImage<TLabelImage3D, TLabelImage3D>(thr_tail, 10, 1);
  m_Data->tp_data[m_PParam.refTP].seg_srs = PTools::
      template Resample3DImage<TLabelImage3D>(dlt_tail, 0.5, ResampleInterpolationMode::NearestNeighbor);

  // Create object name
  m_Data->tp_data[m_PParam.refTP].seg_srs->
      SetObjectName(GenerateUnaryTPObjectName("mask_", m_PParam.refTP, nullptr, "_srs"));
}

template<typename TReal>
void
PropagationAPI<TReal>
::PrepareTimePointData()
{
  m_StdOut->printf("-- [Propagation] Preparing Time Point Data \n");

  std::vector<unsigned int> tps(m_PParam.targetTPs);
  if (std::find(tps.begin(), tps.end(), m_PParam.refTP) == tps.end())
    tps.push_back(m_PParam.refTP); // Add refTP to the tps to process

  for (size_t tp : tps)
		{
		TimePointData<TReal>tpData;

		// Extract full res image
    tpData.img = PTools::template ExtractTimePointImage<TImage3D, TImage4D>(m_Data->img4d, tp);
		tpData.img->SetObjectName(GenerateUnaryTPObjectName("img_", tp));

		// Generate resampled image
    tpData.img_srs = PTools::template Resample3DImage<TImage3D>(tpData.img, 0.5, ResampleInterpolationMode::Linear, 1);
    m_Data->tp_data[tp] = tpData;
		tpData.img_srs->SetObjectName(GenerateUnaryTPObjectName("img_", tp, nullptr, "_srs"));
		}

  // Reference TP Segmentation
  m_Data->tp_data[m_PParam.refTP].seg = m_Data->seg_ref;
  m_Data->tp_data[m_PParam.refTP].seg->SetObjectName(GenerateUnaryTPObjectName("seg_", m_PParam.refTP));
  m_Data->tp_data[m_PParam.refTP].seg_mesh = PTools::GetMeshFromLabelImage(m_Data->seg_ref);

  // Write out the reference mesh
  if (m_PParam.writeOutputToDisk)
    {
    WriteMesh(m_Data->tp_data[m_PParam.refTP].seg_mesh,
        GenerateUnaryTPFileName(m_PParam.fnmeshout_pattern.c_str(), m_PParam.refTP,
                                  m_PParam.outdir.c_str(), ".vtk").c_str());
    }

  ValidateInputOrientation();
  CreateReferenceMask();

	// Debug: write out extracted tp images
  if (m_PParam.debug)
		{
    for (auto &kv : m_Data->tp_data)
			{
      PTools::template WriteImage<TImage3D>(kv.second.img,
          GenerateUnaryTPObjectName("img_", kv.first, m_PParam.debug_dir.c_str(), nullptr, ".nii.gz"));

      PTools::template WriteImage<TImage3D>(kv.second.img_srs,
          GenerateUnaryTPObjectName("img_", kv.first, m_PParam.debug_dir.c_str(), "_srs", ".nii.gz"));
			}

    PTools::template WriteImage<TLabelImage3D>(m_Data->tp_data[m_PParam.refTP].seg_srs,
        GenerateUnaryTPObjectName("mask_", m_PParam.refTP, m_PParam.debug_dir.c_str(), "_srs", ".nii.gz"));
		}
}

template<typename TReal>
void
PropagationAPI<TReal>
::CreateTimePointLists()
{
  m_ForwardTPs.push_back(m_PParam.refTP);
  m_BackwardTPs.push_back(m_PParam.refTP);

  for (unsigned int tp : m_PParam.targetTPs)
		{
    if (tp < m_PParam.refTP)
      m_BackwardTPs.push_back(tp);
    else if (tp == m_PParam.refTP)
			continue; // ignore reference tp in the target list
		else
      m_ForwardTPs.push_back(tp);
		}

  std::sort(m_ForwardTPs.begin(), m_ForwardTPs.end());
  std::sort(m_BackwardTPs.rbegin(), m_BackwardTPs.rend()); // sort backward reversely
}

template <typename TReal>
void
PropagationAPI<TReal>
::RunUnidirectionalPropagation(const std::vector<unsigned int> &tp_list)
{

  m_StdOut->printf("-- [Propagation] Unidirectional Propagation for tp_list: ");
  for (auto tp : tp_list)
    m_StdOut->printf(" %d", tp);
  m_StdOut->printf("\n");

  RunDownSampledPropagation(tp_list); // Generate affine matrices and masks
  GenerateFullResolutionMasks(tp_list); // Reslice downsampled masks to full-res
  GenerateReferenceSpace(tp_list); // Generate reference space for faster run

	// Run reg between ref and target tp and warp reference segmentation to target segmentation
	for (size_t crnt = 1; crnt < tp_list.size(); ++crnt) // this can be parallelized
		{
    RunFullResolutionPropagation(tp_list[crnt]);
		}
}

template <typename TReal>
void
PropagationAPI<TReal>
::RunFullResolutionPropagation(const unsigned int target_tp)
{
  m_StdOut->printf("-- [Propagation] Running Full Resolution Propagation from %02d to %02d\n",
                   m_PParam.refTP, target_tp);

	// Run Deformable Reg from target to ref
  RunPropagationDeformable(target_tp, m_PParam.refTP, true);

	// Warp ref segmentation to target
  RunPropagationReslice(m_PParam.refTP, target_tp, true);

  // Warp ref segmentation mesh to target
  RunPropagationMeshReslice(m_PParam.refTP, target_tp);
}

template <typename TReal>
void
PropagationAPI<TReal>
::GenerateReferenceSpace(const std::vector<unsigned int> &tp_list)
{
  // Add full-resolution masks together for trimming
  using TAddFilter = itk::AddImageFilter<TLabelImage3D, TLabelImage3D, TLabelImage3D>;
  auto fltAdd = TAddFilter::New();
  fltAdd->SetInput1(m_Data->tp_data[tp_list[0]].full_res_mask);
  fltAdd->SetInput2(m_Data->tp_data[tp_list[1]].full_res_mask);
  fltAdd->Update();
  auto img_tail = fltAdd->GetOutput();

  for (size_t i = 2; i < tp_list.size(); ++i)
    {
    fltAdd->SetInput1(img_tail);
    fltAdd->SetInput2(m_Data->tp_data[tp_list[i]].full_res_mask);
    fltAdd->Update();
    img_tail = fltAdd->GetOutput();
    }

  typename TLabelImage3D::RegionType roi;
  auto trimmed = PTools::TrimLabelImage(img_tail, 5, roi);

  // Move trimmed image to the roi region
  trimmed->SetRegions(roi);
  typename TLabelImage3D::PointType origin;
  for (int i = 0; i < 3; ++i)
    {
    origin.SetElement(i, roi.GetIndex().GetElement(i));
    }
  trimmed->SetOrigin(origin);

  auto ref_space = PTools::CastLabelToRealImage(trimmed);

  m_Data->full_res_ref_space = ref_space;

  if (m_PParam.debug)
    {
    std::ostringstream fnrs;
    fnrs << m_PParam.debug_dir << PTools::GetPathSeparator()
         << "full_res_reference_space.nii.gz";
    PTools::template WriteImage<TImage3D>(ref_space, fnrs.str());
    }
}

template <typename TReal>
void
PropagationAPI<TReal>
::GenerateFullResolutionMasks(const std::vector<unsigned int> &tp_list)
{

  m_StdOut->printf("-- [Propagation] Generating Full Resolution Masks \n");

	for (size_t i = 0; i < tp_list.size(); ++i)
		{
		const unsigned int tp = tp_list[i];
    TimePointData<TReal> &tp_data = m_Data->tp_data[tp];
		tp_data.full_res_mask = PropagationTools<TReal>
				::ResliceLabelImageWithIdentityMatrix(tp_data.img, tp_data.seg_srs);
		std::string fnmask = GenerateUnaryTPObjectName("mask_", tp);
    if (m_PParam.debug)
			{
      fnmask = GenerateUnaryTPObjectName("mask_", tp, m_PParam.debug_dir.c_str(), nullptr, ".nii.gz");
      PTools::template WriteImage<TLabelImage3D>(tp_data.full_res_mask, fnmask);
			}
		tp_data.full_res_mask->SetObjectName(fnmask);
		}
}

template<typename TReal>
void
PropagationAPI<TReal>
::RunDownSampledPropagation(const std::vector<unsigned int> &tp_list)
{
  m_StdOut->printf("-- [Propagation] Down Sampled Propagation started  \n");

	for (size_t i = 1; i < tp_list.size(); ++i)
		{
		unsigned int c = tp_list[i], p = tp_list[i - 1]; // current tp and previous tp
    RunPropagationAffine(p, c); // affine reg current to prev
    RunPropagationDeformable(p, c, false); // deformable reg current to prev
    BuildTransformChainForReslice(p, c); // build transformation chain for current tp
    RunPropagationReslice(m_PParam.refTP, c, false); // warp ref mask to current
		}
}

template<typename TReal>
void
PropagationAPI<TReal>
::RunPropagationAffine(unsigned int tp_fix, unsigned int tp_mov)
{
  m_StdOut->printf("-- [Propagation] Running Affine %02d to %02d  \n", tp_mov, tp_fix);
  TimePointData<TReal> &df = m_Data->tp_data[tp_fix], &dm = m_Data->tp_data[tp_mov];

	// Create a new GreedyAPI for affine run and configure
  std::shared_ptr<GreedyApproach<3u, TReal>> GreedyAPI = std::make_shared<GreedyApproach<3u, TReal>>();

	GreedyInputGroup ig;
	ImagePairSpec ip;
	ip.weight = 1.0;

	auto img_fix = df.img_srs;
	auto img_mov = dm.img_srs;

	ip.fixed = img_fix->GetObjectName();
	ip.moving = img_mov->GetObjectName();
	ig.inputs.push_back(ip);

  typename TCompositeImage3D::Pointer casted_fix = PTools::
			CastImageToCompositeImage(img_fix);
  typename TCompositeImage3D::Pointer casted_mov = PTools::
			CastImageToCompositeImage(img_mov);

	GreedyAPI->AddCachedInputObject(ip.fixed, casted_fix);
	GreedyAPI->AddCachedInputObject(ip.moving, casted_mov);

	// Set dilated fix seg as mask
	auto mask_fix = df.seg_srs;
	ig.fixed_mask = mask_fix->GetObjectName();
  auto casted_mask = PTools::CastLabelToRealImage(mask_fix);
	GreedyAPI->AddCachedInputObject(ig.fixed_mask, casted_mask);

	// Configure greedy parameters
	GreedyParameters param;
	param.mode = GreedyParameters::AFFINE;
  param.CopyGeneralSettings(m_GParam); // copy general settings from user input
  param.CopyAffineSettings(m_GParam); // copy affine settings from user input
	// Override global default settings with propagation specific setting
	param.affine_init_mode = AffineInitMode::RAS_IDENTITY;
	param.affine_dof = GreedyParameters::DOF_RIGID;

	// Check smoothing parameters. If greedy default detected, change to propagation default.
	const SmoothingParameters prop_default_pre = { 3.0, true }, prop_default_post = { 1.5, true };
  param.sigma_pre = (m_GParam.sigma_pre == GreedyParameters::default_sigma_pre) ? prop_default_pre : m_GParam.sigma_pre;
  param.sigma_post = (m_GParam.sigma_post == GreedyParameters::default_sigma_post) ? prop_default_post : m_GParam.sigma_post;

	// Add the input group to the parameters
	param.input_groups.clear();
	param.input_groups.push_back(ig);

	// Configure output
	bool force_write = false;
	param.output = GenerateBinaryTPObjectName("affine_", tp_mov, tp_fix);

  if (m_PParam.debug)
		{
		force_write = true;
		param.output = GenerateBinaryTPObjectName("affine_", tp_mov, tp_fix,
                                              m_PParam.debug_dir.c_str(), ".mat");
		}

  m_StdOut->printf("-- [Propagation] Affine Command: %s \n",  param.GenerateCommandLine().c_str());

	dm.affine_to_prev->SetObjectName(param.output);
	GreedyAPI->AddCachedOutputObject(param.output, dm.affine_to_prev, force_write);

	int ret = GreedyAPI->RunAffine(param);

  if (ret != 0)
    throw GreedyException("GreedyAPI execution failed in Proapgation Affine Run: tp_fix = %d, tp_mov = %d",
                          tp_fix, tp_mov);
}

template<typename TReal>
void
PropagationAPI<TReal>
::RunPropagationDeformable(unsigned int tp_fix, unsigned int tp_mov, bool isFullRes)
{
  m_StdOut->printf("-- [Propagation] Running %s Deformable %02d to %02d \n",
                   isFullRes ? "Full-resolution" : "Down-sampled", tp_mov, tp_fix);

	// Get relevant tp data
  TimePointData<TReal> &tpdata_fix = m_Data->tp_data[tp_fix];
  TimePointData<TReal> &tpdata_mov = m_Data->tp_data[tp_mov];

	// Set greedy parameters
  std::shared_ptr<GreedyApproach<3u, TReal>> GreedyAPI = std::make_shared<GreedyApproach<3u, TReal>>();
	GreedyParameters param;
	param.mode = GreedyParameters::GREEDY;
  param.CopyDeformableSettings(m_GParam);
  param.CopyGeneralSettings(m_GParam);

	// Set input images
	GreedyInputGroup ig;
	ImagePairSpec ip;
	ip.weight = 1.0;
	auto img_fix = isFullRes ? tpdata_fix.img : tpdata_fix.img_srs;
	auto img_mov = isFullRes ? tpdata_mov.img : tpdata_mov.img_srs;
	ip.fixed = img_fix->GetObjectName();
	ip.moving = img_mov->GetObjectName();

  typename TCompositeImage3D::Pointer casted_fix = PTools::CastImageToCompositeImage(img_fix);
  typename TCompositeImage3D::Pointer casted_mov = PTools::CastImageToCompositeImage(img_mov);
	GreedyAPI->AddCachedInputObject(ip.fixed, casted_fix);
	GreedyAPI->AddCachedInputObject(ip.moving, casted_mov);
	ig.inputs.push_back(ip);

	// Set mask images
	auto mask_fix = isFullRes ? tpdata_fix.full_res_mask : tpdata_fix.seg_srs;
	ig.fixed_mask = mask_fix->GetObjectName();
  auto casted_mask = PTools::CastLabelToRealImage(mask_fix);
	GreedyAPI->AddCachedInputObject(ig.fixed_mask, casted_mask);

	// Check smoothing parameters. If greedy default detected, change to propagation default.
	const SmoothingParameters prop_default_pre = { 3.0, true }, prop_default_post = { 1.5, true };
  param.sigma_pre = (m_GParam.sigma_pre == GreedyParameters::default_sigma_pre) ? prop_default_pre : m_GParam.sigma_pre;
  param.sigma_post = (m_GParam.sigma_post == GreedyParameters::default_sigma_post) ? prop_default_post : m_GParam.sigma_post;

	// Configure output
	bool force_write = false; // Write out images for debugging
	const char *suffix = isFullRes ? "" : "_srs";
	param.output = GenerateBinaryTPObjectName("warp_", tp_mov, tp_fix, nullptr, suffix);
	param.inverse_warp = GenerateBinaryTPObjectName("warp_", tp_fix, tp_mov, nullptr, suffix);

  if (m_PParam.debug)
		{
		force_write = true;
		param.output = GenerateBinaryTPObjectName("warp_", tp_mov, tp_fix,
        m_PParam.debug_dir.c_str(), suffix, ".nii.gz");
		param.inverse_warp = GenerateBinaryTPObjectName("warp_", tp_fix, tp_mov,
        m_PParam.debug_dir.c_str(), suffix, ".nii.gz");
		}

	using LDDMM3DType = LDDMMData<TReal, 3>;

	if (isFullRes)
		{
		// Set reference space for full res mode
//    param.reference_space = "Full_Resolution_Ref_Space";
//    GreedyAPI->AddCachedInputObject(param.reference_space, m_Data->full_res_ref_space);

		// Set the transformation chain
		for (size_t i = 0; i < tpdata_fix.transform_specs.size(); ++i)
			{
			auto &trans_spec = tpdata_fix.transform_specs[i];
			std::string affine_id = trans_spec.affine->GetObjectName();
			ig.moving_pre_transforms.push_back(TransformSpec(affine_id, -1.0));
			GreedyAPI->AddCachedInputObject(affine_id, trans_spec.affine.GetPointer());
			}

		// Set output objects
		tpdata_fix.deform_from_ref = LDDMM3DType::new_vimg(tpdata_fix.img);
		tpdata_fix.deform_from_ref->SetObjectName(param.output);
		GreedyAPI->AddCachedOutputObject(param.output, tpdata_fix.deform_from_ref, force_write);

		tpdata_fix.deform_to_ref = LDDMM3DType::new_vimg(tpdata_fix.img);
		tpdata_fix.deform_to_ref->SetObjectName(param.inverse_warp);
		GreedyAPI->AddCachedOutputObject(param.inverse_warp, tpdata_fix.deform_to_ref, force_write);
		}
	else
		{
		// Set Initial affine transform
		std::string it_name = tpdata_mov.affine_to_prev->GetObjectName();
		ig.moving_pre_transforms.push_back(TransformSpec(it_name, 1.0));
		GreedyAPI->AddCachedInputObject(it_name, tpdata_mov.affine_to_prev);

		// Set output objects
		tpdata_mov.deform_to_prev = LDDMM3DType::new_vimg(tpdata_mov.img_srs);
		tpdata_mov.deform_to_prev->SetObjectName(param.output);
		GreedyAPI->AddCachedOutputObject(param.output, tpdata_mov.deform_to_prev, force_write);

		tpdata_mov.deform_from_prev = LDDMM3DType::new_vimg(tpdata_mov.img_srs);
		tpdata_mov.deform_from_prev->SetObjectName(param.inverse_warp);
		GreedyAPI->AddCachedOutputObject(param.inverse_warp, tpdata_mov.deform_from_prev, force_write);
		}

	// Add the input group to the parameters
	param.input_groups.clear();
	param.input_groups.push_back(ig);

  m_StdOut->printf("-- [Propagation] Deformable Command: %s \n", param.GenerateCommandLine().c_str());

	int ret = GreedyAPI->RunDeformable(param);

  if (ret != 0)
    throw GreedyException("GreedyAPI execution failed in Proapgation Deformable Run: tp_fix = %d, tp_mov = %d, isFulRes = %d",
                          tp_fix, tp_mov, isFullRes);
}

template<typename TReal>
void
PropagationAPI<TReal>
::RunPropagationReslice(unsigned int tp_in, unsigned int tp_out, bool isFullRes)
{
  m_StdOut->printf("-- [Propagation] Running %s Reslice %02d to %02d \n",
                   isFullRes ? "Full-resolution" : "Down-sampled", tp_in, tp_out);

  TimePointData<TReal> &tpdata_in = m_Data->tp_data[tp_in];
  TimePointData<TReal> &tpdata_out = m_Data->tp_data[tp_out];

	// API and parameter configuration
	GreedyApproach<3u, TReal> *GreedyAPI = new GreedyApproach<3u, TReal>();
	GreedyParameters param;
	param.mode = GreedyParameters::RESLICE;
  param.CopyGeneralSettings(m_GParam);
  param.CopyReslicingSettings(m_GParam);

	// Set reference image
	auto img_ref = isFullRes ? tpdata_out.img : tpdata_out.img_srs;
	param.reslice_param.ref_image = img_ref->GetObjectName();
  auto casted_ref = PTools::CastImageToCompositeImage(img_ref);
	GreedyAPI->AddCachedInputObject(param.reslice_param.ref_image, casted_ref.GetPointer());

	// Set input image
	auto img_in = isFullRes ? tpdata_in.seg : tpdata_in.seg_srs;
	std::string imgin_name = img_in->GetObjectName();
	auto casted_mov = PropagationTools<TReal>
			::template CastToCompositeImage<TLabelImage3D, IdentityIntensityMapping<TReal>>(img_in);
	GreedyAPI->AddCachedInputObject(imgin_name, casted_mov.GetPointer());

	// Set output image
	std::string imgout_name;
	bool force_write = false;
	if (isFullRes)
		{
    force_write = m_PParam.writeOutputToDisk;
    imgout_name = GenerateUnaryTPFileName(m_PParam.fnsegout_pattern.c_str(),
                                          tp_out, m_Data->outdir.c_str(), ".nii.gz");
		}
  else if (m_PParam.debug)
		{
		force_write = true;
    imgout_name = GenerateUnaryTPObjectName("mask_", tp_out, m_PParam.debug_dir.c_str(), "_srs", ".nii.gz");
		}
	else // non debug, non full-res
		{
		imgout_name = GenerateUnaryTPObjectName("mask_", tp_out, nullptr, "_srs");
		}

	auto img_out = TLabelImage3D::New(); // create a new empty image
	img_out->SetObjectName(imgout_name);
	if (isFullRes)
		tpdata_out.seg = img_out.GetPointer();
	else
		tpdata_out.seg_srs = img_out.GetPointer();
	GreedyAPI->AddCachedOutputObject(imgout_name, img_out.GetPointer(), force_write);

	// Make a reslice spec with input-output pair and push to the parameter
  ResliceSpec rspec(imgin_name, imgout_name, m_PParam.reslice_spec);
	param.reslice_param.images.push_back(rspec);

	// Build transformation chain
	if (isFullRes)
		{
		// Prepend deformation field before all affine matrices for full-res reslice
		std::string deform_id = tpdata_out.deform_from_ref->GetObjectName();
		param.reslice_param.transforms.push_back(TransformSpec(deform_id));
		GreedyAPI->AddCachedInputObject(deform_id, tpdata_out.deform_from_ref.GetPointer());
		}

	for (size_t i = 0; i < tpdata_out.transform_specs.size(); ++i)
		{
		auto &trans_spec = tpdata_out.transform_specs[i];
		std::string affine_id = trans_spec.affine->GetObjectName();
		param.reslice_param.transforms.push_back(TransformSpec(affine_id, -1.0));
		GreedyAPI->AddCachedInputObject(affine_id, trans_spec.affine.GetPointer());

		// Append deformation field to each affine matrix for downsampled propagation
		if (!isFullRes)
			{
			std::string deform_id = trans_spec.deform->GetObjectName();
			param.reslice_param.transforms.push_back(TransformSpec(deform_id));
			GreedyAPI->AddCachedInputObject(deform_id, trans_spec.deform.GetPointer());
			}
		}

  m_StdOut->printf("-- [Propagation] Reslice Command: %s \n", param.GenerateCommandLine().c_str());

	int ret = GreedyAPI->RunReslice(param);

  if (ret != 0)
    throw GreedyException("GreedyAPI execution failed in Proapgation Reslice Run: tp_in = %d, tp_out = %d, isFulRes = %d",
                          tp_in, tp_out, isFullRes);
}

template<typename TReal>
void
PropagationAPI<TReal>
::RunPropagationMeshReslice(unsigned int tp_in, unsigned int tp_out)
{
  m_StdOut->printf("-- [Propagation] Running Mesh Reslice %02d to %02d \n", tp_in, tp_out);

  TimePointData<TReal> &tpdata_in = m_Data->tp_data[tp_in];
  TimePointData<TReal> &tpdata_out = m_Data->tp_data[tp_out];

  // API and parameter configuration
  GreedyApproach<3u, TReal> *GreedyAPI = new GreedyApproach<3u, TReal>();
  GreedyParameters param;
  param.mode = GreedyParameters::RESLICE;
  param.CopyGeneralSettings(m_GParam);
  param.CopyReslicingSettings(m_GParam);

  // Set reference image
  auto img_ref = tpdata_out.img;
  param.reslice_param.ref_image = img_ref->GetObjectName();
  auto casted_ref = PTools::CastImageToCompositeImage(img_ref);
  GreedyAPI->AddCachedInputObject(param.reslice_param.ref_image, casted_ref.GetPointer());

  // Set input mesh
  auto mesh_in = tpdata_in.seg_mesh;
  std::string mesh_in_name = GenerateUnaryTPObjectName("mesh_", tp_in, nullptr, nullptr, ".vtk");
  GreedyAPI->AddCachedInputObject(mesh_in_name, mesh_in);

  // Set output image
  std::string mesh_out_name =
      GenerateUnaryTPFileName(m_PParam.fnmeshout_pattern.c_str(), tp_out, m_PParam.outdir.c_str(), ".vtk");

  // Make a reslice spec with input-output pair and push to the parameter
  ResliceMeshSpec rmspec(mesh_in_name, mesh_out_name);
  param.reslice_param.meshes.push_back(rmspec);
  tpdata_out.seg_mesh = TPropagationMesh::New();
  GreedyAPI->AddCachedOutputObject(mesh_out_name, tpdata_out.seg_mesh, m_PParam.writeOutputToDisk);

  // Add extra meshes to warp
  for (auto &mesh_spec : m_PParam.extra_mesh_list)
    {
    ResliceMeshSpec rms;

    if (mesh_spec.cached)
      {
      auto tag = mesh_spec.fnout_pattern;
      rms.fixed = tag;

      // add input to cache
      auto mesh_in = tpdata_in.extra_meshes[tag];
      GreedyAPI->AddCachedInputObject(tag, mesh_in);

      // configure output
      tpdata_out.extra_meshes[tag] = TPropagationMesh::New();
      std::string out_name = GenerateBinaryTPObjectName(tag.c_str(), tp_in, tp_out,
                                                        nullptr, nullptr, nullptr);
      GreedyAPI->AddCachedOutputObject(out_name, tpdata_out.extra_meshes[tag], false);
      }
    else
      {
      auto pattern = mesh_spec.fnout_pattern;
      rms.fixed = mesh_spec.fn_mesh;
      std::string fn_mesh_ref = GenerateUnaryTPFileName(pattern.c_str(), m_PParam.refTP,
                                                        m_PParam.outdir.c_str(), ".vtk");
      itksys::SystemTools::CopyAFile(mesh_spec.fn_mesh, fn_mesh_ref);
      rms.output = GenerateUnaryTPFileName(pattern.c_str(), tp_out, m_PParam.outdir.c_str(), ".vtk");
      }

    param.reslice_param.meshes.push_back(rms);
    }

  // Build transformation chain
  for (int i = tpdata_out.transform_specs.size() - 1; i >= 0; --i)
    {
    auto &trans_spec = tpdata_out.transform_specs[i];
    std::string affine_id = trans_spec.affine->GetObjectName();
    param.reslice_param.transforms.push_back(TransformSpec(affine_id));
    GreedyAPI->AddCachedInputObject(affine_id, trans_spec.affine.GetPointer());
    }

  std::string deform_id = tpdata_out.deform_to_ref->GetObjectName();
  param.reslice_param.transforms.push_back(TransformSpec(deform_id));
  GreedyAPI->AddCachedInputObject(deform_id, tpdata_out.deform_to_ref.GetPointer());

  m_StdOut->printf("-- [Propagation] Mesh Reslice Command: %s \n", param.GenerateCommandLine().c_str());

  int ret = GreedyAPI->RunReslice(param);

  if (ret != 0)
    throw GreedyException("GreedyAPI execution failed in Proapgation Mesh Reslice Run: tp_in = %d, tp_out = %d",
                          tp_in, tp_out);
}

template<typename TReal>
void
PropagationAPI<TReal>
::BuildTransformChainForReslice(unsigned int tp_prev, unsigned int tp_crnt)
{
  m_StdOut->printf("-- [Propagation] Building reslicing transformation chain for tp: %02d\n", tp_crnt);

  TimePointData<TReal> &tpdata_crnt = m_Data->tp_data[tp_crnt];
  TimePointData<TReal> &tpdata_prev = m_Data->tp_data[tp_prev];

	// Copy previous transform specs as a starting point
	for (auto &spec : tpdata_prev.transform_specs)
		tpdata_crnt.transform_specs.push_back(spec);

	// Get current transformations
	auto affine = tpdata_crnt.affine_to_prev;
	auto deform = tpdata_crnt.deform_from_prev;

	// Build spec and append to existing list
	TimePointTransformSpec<TReal> spec(affine, deform, tp_crnt);
	tpdata_crnt.transform_specs.push_back(spec);
}

template<typename TReal>
void
PropagationAPI<TReal>
::Generate4DSegmentation()
{
  auto fltJoin = itk::JoinSeriesImageFilter<TLabelImage3D, TLabelImage4D>::New();

  for (size_t i = 1; i <= m_Data->GetNumberOfTimePoints(); ++i)
    {
    if (m_Data->tp_data.count(i))
      {
      // Append slice to the list
      fltJoin->PushBackInput(m_Data->tp_data[i].seg);
      }
    else
      {
      // Append an empty image
      auto refseg = m_Data->tp_data[m_PParam.refTP].seg;
      auto emptyImage = PTools::template CreateEmptyImage<TLabelImage3D>(refseg);
      fltJoin->PushBackInput(emptyImage);
      }
    }
  fltJoin->Update();
  m_Data->seg4d_out = fltJoin->GetOutput();

  if (m_PParam.writeOutputToDisk)
  {
    std::ostringstream fnseg4d;
    fnseg4d << m_PParam.outdir<< PTools::GetPathSeparator()
            << "seg4d.nii.gz";
    PTools::template WriteImage<TLabelImage4D>(m_Data->seg4d_out, fnseg4d.str());
  }
}

template <typename TReal>
inline std::string
PropagationAPI<TReal>
::GenerateUnaryTPObjectName(const char *base, unsigned int tp,
														const char *debug_dir, const char *suffix, const char *file_ext)
{
	std::ostringstream oss;
	if (debug_dir)
    oss << debug_dir << PTools::GetPathSeparator();
	if (base)
		oss << base;

	oss << setfill('0') << setw(2) << tp;

	if (suffix)
		oss << suffix;
	if (file_ext)
		oss << file_ext;

	return oss.str();
}

template <typename TReal>
inline std::string
PropagationAPI<TReal>
::GenerateBinaryTPObjectName(const char *base, unsigned int tp1, unsigned int tp2,
														const char *debug_dir, const char *suffix, const char *file_ext)
{
	std::ostringstream oss;
	if (debug_dir)
    oss << debug_dir << PTools::GetPathSeparator();
	if (base)
		oss << base;

	oss << setfill('0') << setw(2) << tp1 << "_to_" << setfill('0') << setw(2) << tp2;

	if (suffix)
		oss << suffix;
	if (file_ext)
		oss << file_ext;

	return oss.str();
}

template<typename TReal>
inline std::string
PropagationAPI<TReal>
::GenerateUnaryTPFileName(const char *pattern, unsigned int tp,
                          const char *output_dir, const char *file_ext)
{
  std::ostringstream oss;
  if (output_dir)
    oss << output_dir << PTools::GetPathSeparator();

  if (strchr(pattern, '%') == NULL)
    {
    // % pattern not found, append tp after the pattern
    oss << pattern << "_" << setfill('0') << setw(2) << tp;
    if (file_ext)
      oss << file_ext;
    }
  else
    {
    // use pattern specified by user
    oss << PTools::ssprintf(pattern, tp);
    }

  return oss.str();
}

template<typename TReal>
std::shared_ptr<PropagationOutput<TReal>>
PropagationAPI<TReal>
::GetOutput()
{
  std::shared_ptr<PropagationOutput<TReal>> ret =
      std::make_shared<PropagationOutput<TReal>>();

  ret->Initialize(m_Data);
  return ret;
}

//=================================================
// PropagationStdOut Definition
//=================================================

PropagationStdOut
::PropagationStdOut(PropagationParameters::Verbosity verbosity, FILE *f_out)
: m_Verbosity(verbosity), m_Output(f_out ? f_out : stdout)
{

}

PropagationStdOut
::~PropagationStdOut()
{

}

void
PropagationStdOut
::printf(const char *format, ...)
{
  if(m_Verbosity > PropagationParameters::VERB_NONE)
    {
    char buffer[4096];
    va_list args;
    va_start (args, format);
    vsprintf (buffer,format, args);
    va_end (args);

    fprintf(m_Output, "%s", buffer);
    }
}

void
PropagationStdOut
::print_verbose(const char *format, ...)
{
  if (m_Verbosity == PropagationParameters::VERB_VERBOSE)
    {
    va_list args;
    va_start (args, format);
    this->printf(format, args);
    }
}

void
PropagationStdOut
::flush()
{
  fflush(m_Output);
}

namespace propagation
{
	template class PropagationAPI<float>;
	template class PropagationAPI<double>;
}
