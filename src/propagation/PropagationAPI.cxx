#include "PropagationAPI.h"
#include "GreedyAPI.h"
#include "PropagationTools.h"
#include "PropagationData.h"
#include "PropagationIO.h"

#include <algorithm>

#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>


using namespace propagation;

template<typename TReal>
PropagationAPI<TReal>
::PropagationAPI(const std::shared_ptr<PropagationInput<TReal>> input)
{
  m_Param = input->m_Param;
  m_Data = input->m_Data;
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
  if (m_Param.propagation_param.refTP < 0 || m_Param.propagation_param.refTP > nt)
    throw GreedyException("Reference tp %d cannot be greater than total number of tps %d",
                          m_Param.propagation_param.refTP, nt);

  for (size_t tp : m_Param.propagation_param.targetTPs)
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
	std::cout << "Run Started" << std::endl;

  const auto &pParam = m_Param.propagation_param;
	PropagationData<TReal> pData;

  if (pParam.debug)
		{
		std::cout << "-- [Propagation] Debug Mode is ON" << std::endl;
    std::cout << "-- [Propagation] Debug Output Dir: " << pParam.debug_dir << std::endl;
		}

  PrepareTimePointData(); // Prepare initial timepoint data for propagation
  CreateTimePointLists();

  if (pParam.debug)
		{
		std::cout << "-- [Propagation] forward list: ";
    for (auto tp : m_ForwardTPs)
			std::cout << " " << tp;
		std::cout << std::endl << "-- [Propagation] backward list: ";
    for (auto tp : m_BackwardTPs)
			std::cout << " " << tp;
		std::cout << std::endl;
		}

	// Run forward propagation
  if (m_ForwardTPs.size() > 1)
    RunUnidirectionalPropagation(m_ForwardTPs);

	// Run backward propagation
  if (m_BackwardTPs.size() > 1)
    RunUnidirectionalPropagation(m_BackwardTPs);

	std::cout << "Run Completed" << std::endl;

  return EXIT_SUCCESS;
}

template<typename TReal>
void
PropagationAPI<TReal>
::ValidateInputOrientation()
{
  const auto &pParam = m_Param.propagation_param;
  typename TImage3D::DirectionType img_direction, seg_direction;

  img_direction = m_Data->tp_data[pParam.refTP].img->GetDirection();
  seg_direction = m_Data->seg_ref->GetDirection();

  if (img_direction != seg_direction)
    {
    std::cerr << "Image Direction: " << std::endl << img_direction << std::endl;
    std::cerr << "Segmentation Direction: " << std::endl << seg_direction << std::endl;
    throw GreedyException("Image and Segmentation orientations do not match. Segmentation file %s\n",
                          pParam.segspec.refseg.c_str());
    }
}

template<typename TReal>
void
PropagationAPI<TReal>
::CreateReferenceMask()
{
  const auto &pParam = m_Param.propagation_param;

  // Threshold, Dilate and Resample
  auto thr_tail = PropagationTools<TReal>::template ThresholdImage<TLabelImage3D, TLabelImage3D>(
        m_Data->seg_ref, 1, SHRT_MAX, 1, 0);
  auto dlt_tail = PropagationTools<TReal>::template DilateImage<TLabelImage3D, TLabelImage3D>(
        thr_tail, 10, 1);
  m_Data->tp_data[pParam.refTP].seg_srs = PropagationTools<TReal>::
      template Resample3DImage<TLabelImage3D>(dlt_tail, 0.5, ResampleInterpolationMode::NearestNeighbor);

  // Create object name
  m_Data->tp_data[pParam.refTP].seg_srs->
      SetObjectName(GenerateUnaryTPObjectName("mask_", pParam.refTP, nullptr, "_srs"));
}

template<typename TReal>
void
PropagationAPI<TReal>
::PrepareTimePointData()
{
  const auto &pParam = m_Param.propagation_param;
  if (pParam.debug)
		std::cout << "[Propagation] PrepareTimePointData" << std::endl;

  std::cout << "[Propagation] Processing Time Point Images" << std::endl;

  std::vector<unsigned int> tps(pParam.targetTPs);
  if (std::find(tps.begin(), tps.end(), pParam.refTP) == tps.end())
    tps.push_back(pParam.refTP); // Add refTP to the tps to process

  for (size_t tp : tps)
		{
		std::cout << "-- Processing tp " << tp << std::endl;
		TimePointData<TReal>tpData;

		// Extract full res image
		tpData.img = PropagationTools<TReal>::
        template ExtractTimePointImage<TImage3D, TImage4D>(m_Data->img4d, tp);
		tpData.img->SetObjectName(GenerateUnaryTPObjectName("img_", tp));

		// Generate resampled image
		tpData.img_srs = PropagationTools<TReal>::
				template Resample3DImage<TImage3D>(tpData.img, 0.5, ResampleInterpolationMode::Linear, 1);

    m_Data->tp_data[tp] = tpData;
		tpData.img_srs->SetObjectName(GenerateUnaryTPObjectName("img_", tp, nullptr, "_srs"));
		}

  // Reference TP Segmentation
  m_Data->tp_data[pParam.refTP].seg = m_Data->seg_ref;
  m_Data->tp_data[pParam.refTP].seg->SetObjectName(GenerateUnaryTPObjectName("seg_", pParam.refTP));

  ValidateInputOrientation();
  CreateReferenceMask();

	// Debug: write out extracted tp images
  if (pParam.debug)
		{
    for (auto &kv : m_Data->tp_data)
			{
      PropagationTools<TReal>::template WriteImage<TImage3D>(kv.second.img,
          GenerateUnaryTPObjectName("img_", kv.first, pParam.debug_dir.c_str(), nullptr, ".nii.gz"));

      PropagationTools<TReal>::template WriteImage<TImage3D>(kv.second.img_srs,
          GenerateUnaryTPObjectName("img_", kv.first, pParam.debug_dir.c_str(), "_srs", ".nii.gz"));
			}

		PropagationTools<TReal>::template WriteImage<TLabelImage3D>(m_Data->tp_data[pParam.refTP].seg_srs,
        GenerateUnaryTPObjectName("mask_", pParam.refTP, pParam.debug_dir.c_str(), "_srs", ".nii.gz"));
		}
}

template<typename TReal>
void
PropagationAPI<TReal>
::CreateTimePointLists()
{
  const auto &pParam = m_Param.propagation_param;
  m_ForwardTPs.push_back(pParam.refTP);
  m_BackwardTPs.push_back(pParam.refTP);

  for (unsigned int tp : pParam.targetTPs)
		{
    if (tp < pParam.refTP)
      m_BackwardTPs.push_back(tp);
    else if (tp == pParam.refTP)
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
  const auto &pParam = m_Param.propagation_param;
  if (pParam.debug)
		{
		std::cout << "-- [Propagation] Unidirectional Propagation for tp_list: ";
		for (auto tp : tp_list)
			std::cout << " " << tp;
		std::cout << std::endl;
		}

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
  const auto &pParam = m_Param.propagation_param;
  if (pParam.debug)
		{
		std::cout << "-- [Propagation] Running Full Resolution Propagation from "
              << "reference tp: " << pParam.refTP << " to target tp: " << target_tp << std::endl;
		}

	// Run Deformable Reg from target to ref
  RunPropagationDeformable(target_tp, pParam.refTP, true);

	// Warp ref segmentation to target
  RunPropagationReslice(pParam.refTP, target_tp, true);
}

template <typename TReal>
void
PropagationAPI<TReal>
::GenerateReferenceSpace(const std::vector<unsigned int> &tp_list)
{
}

template <typename TReal>
void
PropagationAPI<TReal>
::GenerateFullResolutionMasks(const std::vector<unsigned int> &tp_list)
{
  const auto &pParam = m_Param.propagation_param;
  if (pParam.debug)
		{
		std::cout << "-- [Propagation] Generating Full Resolution Masks " << std::endl;
		}

	for (size_t i = 0; i < tp_list.size(); ++i)
		{
		const unsigned int tp = tp_list[i];
    TimePointData<TReal> &tp_data = m_Data->tp_data[tp];
		tp_data.full_res_mask = PropagationTools<TReal>
				::ResliceLabelImageWithIdentityMatrix(tp_data.img, tp_data.seg_srs);
		std::string fnmask = GenerateUnaryTPObjectName("mask_", tp);
    if (pParam.debug)
			{
      fnmask = GenerateUnaryTPObjectName("mask_", tp, pParam.debug_dir.c_str(), nullptr, ".nii.gz");
			PropagationTools<TReal>::template WriteImage<TLabelImage3D>(tp_data.full_res_mask, fnmask);
			}
		tp_data.full_res_mask->SetObjectName(fnmask);
		}
}


template<typename TReal>
void
PropagationAPI<TReal>
::RunDownSampledPropagation(const std::vector<unsigned int> &tp_list)
{
  const auto &pParam = m_Param.propagation_param;
  if (pParam.debug)
		{
		std::cout << "-- [Propagation] Down Sampled Propagation started " << std::endl;
		}

	for (size_t i = 1; i < tp_list.size(); ++i)
		{
		unsigned int c = tp_list[i], p = tp_list[i - 1]; // current tp and previous tp
    RunPropagationAffine(p, c); // affine reg current to prev
    RunPropagationDeformable(p, c, false); // deformable reg current to prev
    BuildTransformChainForReslice(p, c); // build transformation chain for current tp
    RunPropagationReslice(pParam.refTP, c, false); // warp ref mask to current
		}
}

template<typename TReal>
void
PropagationAPI<TReal>
::RunPropagationAffine(unsigned int tp_fix, unsigned int tp_mov)
{
  const auto &pParam = m_Param.propagation_param;
  if (pParam.debug)
		{
		std::cout << "-- [Propagation] Running Propagation Affine Started tp_fix=" << tp_fix
							<< "; tp_mov=" << tp_mov << std::endl;
		}

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

	typename TCompositeImage3D::Pointer casted_fix = PropagationTools<TReal>::
			CastImageToCompositeImage(img_fix);
	typename TCompositeImage3D::Pointer casted_mov = PropagationTools<TReal>::
			CastImageToCompositeImage(img_mov);

	GreedyAPI->AddCachedInputObject(ip.fixed, casted_fix);
	GreedyAPI->AddCachedInputObject(ip.moving, casted_mov);

	// Set dilated fix seg as mask
	auto mask_fix = df.seg_srs;
	ig.fixed_mask = mask_fix->GetObjectName();
	auto casted_mask = PropagationTools<TReal>::CastLabelToRealImage(mask_fix);
	GreedyAPI->AddCachedInputObject(ig.fixed_mask, casted_mask);

	// Configure greedy parameters
	GreedyParameters param;
	param.mode = GreedyParameters::AFFINE;
  param.CopyGeneralSettings(m_Param); // copy general settings from user input
  param.CopyAffineSettings(m_Param); // copy affine settings from user input
	// Override global default settings with propagation specific setting
	param.affine_init_mode = AffineInitMode::RAS_IDENTITY;
	param.affine_dof = GreedyParameters::DOF_RIGID;

	// Check smoothing parameters. If greedy default detected, change to propagation default.
	const SmoothingParameters prop_default_pre = { 3.0, true }, prop_default_post = { 1.5, true };
  param.sigma_pre = (m_Param.sigma_pre == GreedyParameters::default_sigma_pre) ? prop_default_pre : m_Param.sigma_pre;
  param.sigma_post = (m_Param.sigma_post == GreedyParameters::default_sigma_post) ? prop_default_post : m_Param.sigma_post;

	// Add the input group to the parameters
	param.input_groups.clear();
	param.input_groups.push_back(ig);

	// Configure output
	bool force_write = false;
	param.output = GenerateBinaryTPObjectName("affine_", tp_mov, tp_fix);

  if (pParam.debug)
		{
		force_write = true;
		param.output = GenerateBinaryTPObjectName("affine_", tp_mov, tp_fix,
                                              pParam.debug_dir.c_str(), ".mat");
		std::cout << "-- [Propagation] Affine Command: " << param.GenerateCommandLine() << std::endl;
		}

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
  const auto &pParam = m_Param.propagation_param;
  if (pParam.debug)
		{
		std::cout << "-- [Propagation] Deformable Run: tp_fix=" << tp_fix
							<< "; tp_mov = " << tp_mov << "; isFullRes=" << isFullRes << std::endl;
		}

	// Get relevant tp data
  TimePointData<TReal> &tpdata_fix = m_Data->tp_data[tp_fix];
  TimePointData<TReal> &tpdata_mov = m_Data->tp_data[tp_mov];

	// Set greedy parameters
  std::shared_ptr<GreedyApproach<3u, TReal>> GreedyAPI = std::make_shared<GreedyApproach<3u, TReal>>();
	GreedyParameters param;
	param.mode = GreedyParameters::GREEDY;
  param.CopyDeformableSettings(m_Param);
  param.CopyGeneralSettings(m_Param);

	// Set input images
	GreedyInputGroup ig;
	ImagePairSpec ip;
	ip.weight = 1.0;
	auto img_fix = isFullRes ? tpdata_fix.img : tpdata_fix.img_srs;
	auto img_mov = isFullRes ? tpdata_mov.img : tpdata_mov.img_srs;
	ip.fixed = img_fix->GetObjectName();
	ip.moving = img_mov->GetObjectName();

	typename TCompositeImage3D::Pointer casted_fix = PropagationTools<TReal>::CastImageToCompositeImage(img_fix);
	typename TCompositeImage3D::Pointer casted_mov = PropagationTools<TReal>::CastImageToCompositeImage(img_mov);
	GreedyAPI->AddCachedInputObject(ip.fixed, casted_fix);
	GreedyAPI->AddCachedInputObject(ip.moving, casted_mov);
	ig.inputs.push_back(ip);

	// Set mask images
	auto mask_fix = isFullRes ? tpdata_fix.full_res_mask : tpdata_fix.seg_srs;
	ig.fixed_mask = mask_fix->GetObjectName();
	auto casted_mask = PropagationTools<TReal>::CastLabelToRealImage(mask_fix);
	GreedyAPI->AddCachedInputObject(ig.fixed_mask, casted_mask);

	// Check smoothing parameters. If greedy default detected, change to propagation default.
	const SmoothingParameters prop_default_pre = { 3.0, true }, prop_default_post = { 1.5, true };
  param.sigma_pre = (m_Param.sigma_pre == GreedyParameters::default_sigma_pre) ? prop_default_pre : m_Param.sigma_pre;
  param.sigma_post = (m_Param.sigma_post == GreedyParameters::default_sigma_post) ? prop_default_post : m_Param.sigma_post;

	// Configure output
	bool force_write = false; // Write out images for debugging
	const char *suffix = isFullRes ? "" : "_srs";
	param.output = GenerateBinaryTPObjectName("warp_", tp_mov, tp_fix, nullptr, suffix);
	param.inverse_warp = GenerateBinaryTPObjectName("warp_", tp_fix, tp_mov, nullptr, suffix);

  if (pParam.debug)
		{
		force_write = true;
		param.output = GenerateBinaryTPObjectName("warp_", tp_mov, tp_fix,
        pParam.debug_dir.c_str(), suffix, ".nii.gz");
		param.inverse_warp = GenerateBinaryTPObjectName("warp_", tp_fix, tp_mov,
        pParam.debug_dir.c_str(), suffix, ".nii.gz");
		}

	using LDDMM3DType = LDDMMData<TReal, 3>;

	if (isFullRes)
		{
		// Set reference space for full res mode
		//param.reference_space = glparam.reference_space;
		//GreedyAPI->AddCachedInputObject(param.reference_space, pData.full_res_ref_space);

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

  if (pParam.debug)
		std::cout << "-- [Propagation] Deformable Command: " <<  param.GenerateCommandLine() << std::endl;

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
  const auto &pParam = m_Param.propagation_param;
  if (pParam.debug)
		{
		std::cout << "-- [Propagation] Reslice Run. tp_in=" << tp_in << "; tp_out="
							<< tp_out << "; isFullRes=" << isFullRes << std::endl;
		}

  TimePointData<TReal> &tpdata_in = m_Data->tp_data[tp_in];
  TimePointData<TReal> &tpdata_out = m_Data->tp_data[tp_out];

	// API and parameter configuration
	GreedyApproach<3u, TReal> *GreedyAPI = new GreedyApproach<3u, TReal>();
	GreedyParameters param;
	param.mode = GreedyParameters::RESLICE;
  param.CopyGeneralSettings(m_Param);
  param.CopyReslicingSettings(m_Param);

	// Set reference image
	auto img_ref = isFullRes ? tpdata_out.img : tpdata_out.img_srs;
	param.reslice_param.ref_image = img_ref->GetObjectName();
	auto casted_ref = PropagationTools<TReal>::CastImageToCompositeImage(img_ref);
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
    force_write = pParam.writeOutputToDisk;
    imgout_name = GenerateUnaryTPObjectName("seg_", tp_out, m_Data->outdir.c_str(), "_resliced", ".nii.gz");
		}
  else if (pParam.debug)
		{
		force_write = true;
    imgout_name = GenerateUnaryTPObjectName("mask_", tp_out, pParam.debug_dir.c_str(), "_srs", ".nii.gz");
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
  ResliceSpec rspec(imgin_name, imgout_name, pParam.reslice_spec);
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

	if (param.propagation_param.debug)
		{
		std::cout << "-- [Propagation] Reslice Command:" << param.GenerateCommandLine() << std::endl;
		}

	int ret = GreedyAPI->RunReslice(param);

  if (ret != 0)
    throw GreedyException("GreedyAPI execution failed in Proapgation Reslice Run: tp_in = %d, tp_out = %d, isFulRes = %d",
                          tp_in, tp_out, isFullRes);
}

template<typename TReal>
void
PropagationAPI<TReal>
::BuildTransformChainForReslice(unsigned int tp_prev, unsigned int tp_crnt)
{
  const auto &pParam = m_Param.propagation_param;
  if (pParam.debug)
		{
		std::cout << "-- [Propagation] Building reslicing trans chain for tp: " << tp_crnt << std::endl;
		}

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

template <typename TReal>
inline std::string
PropagationAPI<TReal>
::GenerateUnaryTPObjectName(const char *base, unsigned int tp,
														const char *debug_dir, const char *suffix, const char *file_ext)
{
	std::ostringstream oss;
	if (debug_dir)
		oss << debug_dir << PropagationTools<TReal>::GetPathSeparator();
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
		oss << debug_dir << PropagationTools<TReal>::GetPathSeparator();
	if (base)
		oss << base;

	oss << setfill('0') << setw(2) << tp1 << "_to_" << setfill('0') << setw(2) << tp2;

	if (suffix)
		oss << suffix;
	if (file_ext)
		oss << file_ext;

	return oss.str();
}

namespace propagation
{
	template class PropagationAPI<float>;
	template class PropagationAPI<double>;
}