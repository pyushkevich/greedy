#include "PropagationAPI.h"
#include "GreedyAPI.h"
#include "PropagationTools.h"
#include "PropagationData.h"

#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>



using namespace propagation;

template<typename TReal>
PropagationAPI<TReal>
::PropagationAPI()
{

}

template<typename TReal>
PropagationAPI<TReal>
::~PropagationAPI()
{

}

template<typename TReal>
int
PropagationAPI<TReal>
::Run(const GreedyParameters &param)
{
	std::cout << "Run Started" << std::endl;
	const GreedyPropagationParameters propa_param = param.propagation_param;
	PropagationData<TReal> pData;
	std::vector<unsigned int> forward_tps;
	std::vector<unsigned int> backward_tps;

	if (propa_param.debug)
		{
		std::cout << "-- [Propagation] Debug Mode is ON" << std::endl;
		std::cout << "-- [Propagation] Debug Output Dir: " << propa_param.debug_dir << std::endl;
		}

	PrepareTimePointData(pData, propa_param); // Prepare initial timepoint data for propagation
	CreateTimePointLists(propa_param.targetTPs, forward_tps, backward_tps, propa_param.refTP);

	if (propa_param.debug)
		{
		std::cout << "-- [Propagation] forward list: ";
		for (auto tp : forward_tps)
			std::cout << " " << tp;
		std::cout << std::endl << "-- [Propagation] backward list: ";
		for (auto tp : backward_tps)
			std::cout << " " << tp;
		std::cout << std::endl;
		}

	// Run forward propagation
	if (forward_tps.size() > 1)
		RunUnidirectionalPropagation(pData, param, forward_tps);

	// Run backward propagation
	if (backward_tps.size() > 1)
		RunUnidirectionalPropagation(pData, param, backward_tps);

	std::cout << "Run Completed" << std::endl;
	return EXIT_SUCCESS;
}

template<typename TReal>
void
PropagationAPI<TReal>
::PrepareTimePointData(PropagationData<TReal> &propa_data, const GreedyPropagationParameters &propa_param)
{
	if (propa_param.debug)
		std::cout << "[Propagation] PrepareTimePointData" << std::endl;

	// Read image 4d
	propa_data.img4d = PropagationTools<TReal>::template ReadImage<TImage4D>(propa_param.img4d);
	auto nt = propa_data.img4d->GetBufferedRegion().GetSize()[3];

	// Validate inputs
	if (propa_param.refTP < 0 || propa_param.refTP > nt)
		throw GreedyException("Reference tp %d cannot be greater than total number of tps %d", propa_param.refTP, nt);
	for (size_t tp : propa_param.targetTPs)
		if (tp > nt) throw GreedyException("Target tp %d cannot be greater than total number tps %d",tp, nt);

	// Set images for the reference TP
	propa_data.tp_data[propa_param.refTP].img = PropagationTools<TReal>::
			template ExtractTimePointImage<TImage3D, TImage4D>(propa_data.img4d, propa_param.refTP);
	propa_data.tp_data[propa_param.refTP].img->SetObjectName(GenerateUnaryTPObjectName("img_", propa_param.refTP));

	propa_data.tp_data[propa_param.refTP].img_srs = PropagationTools<TReal>::
			template Resample3DImage<TImage3D>(propa_data.tp_data[propa_param.refTP].img,0.5,ResampleInterpolationMode::Linear, 1);
	propa_data.tp_data[propa_param.refTP].img_srs->SetObjectName(GenerateUnaryTPObjectName("img_", propa_param.refTP, nullptr, "_srs"));

	std::cout << "[Propagation] Processing Time Point Images" << std::endl;
	// Process timepoint images
	for (size_t tp : propa_param.targetTPs)
		{
		std::cout << "-- Processing tp " << tp << std::endl;
		TimePointData<TReal>tpData;

		// Extract full res image
		tpData.img = PropagationTools<TReal>::
				template ExtractTimePointImage<TImage3D, TImage4D>(propa_data.img4d, tp);
		tpData.img->SetObjectName(GenerateUnaryTPObjectName("img_", tp));

		// Generate resampled image
		tpData.img_srs = PropagationTools<TReal>::
				template Resample3DImage<TImage3D>(tpData.img, 0.5, ResampleInterpolationMode::Linear, 1);
		propa_data.tp_data[tp] = tpData;
		tpData.img_srs->SetObjectName(GenerateUnaryTPObjectName("img_", tp, nullptr, "_srs"));
		}

	typename TImage3D::DirectionType img_direction, seg_direction;
	img_direction = propa_data.tp_data[propa_param.refTP].img->GetDirection();

	// Process segmentations
	for (auto it : propa_param.segpair)
		{
		PropagationSegGroup<TReal> segGroup;
		segGroup.seg_ref = PropagationTools<TReal>::template ReadImage<TLabelImage3D>(it.refseg);
		segGroup.outdir = it.outsegdir;
		segGroup.seg_ref->SetObjectName(GenerateUnaryTPObjectName("seg_", propa_param.refTP));

		seg_direction = segGroup.seg_ref->GetDirection();

		if (img_direction != seg_direction)
			{
			std::cerr << "Image Direction: " << std::endl << img_direction << std::endl;
			std::cerr << "Segmentation Direction: " << std::endl << seg_direction << std::endl;
			throw GreedyException("Image and Segmentation orientations do not match. Segmentation file %s\n",
														it.refseg.c_str());
			}

		// Threshold, Dilate and Resample
		auto thr_tail = PropagationTools<TReal>::template ThresholdImage<TLabelImage3D, TLabelImage3D>(
					segGroup.seg_ref, 1, SHRT_MAX, 1, 0);
		auto dlt_tail = PropagationTools<TReal>::template DilateImage<TLabelImage3D, TLabelImage3D>(
					thr_tail, 10, 1);
		segGroup.seg_ref_srs = PropagationTools<TReal>::
				template Resample3DImage<TLabelImage3D>(dlt_tail, 0.5, ResampleInterpolationMode::NearestNeighbor);
		segGroup.seg_ref_srs->SetObjectName(GenerateUnaryTPObjectName("mask_", propa_param.refTP, nullptr, "_srs"));

		// Generating mesh data
		segGroup.mesh_ref = PropagationTools<TReal>::GetMeshFromLabelImage(segGroup.seg_ref);

		propa_data.seg_list.push_back(segGroup);
		}

	propa_data.tp_data[propa_param.refTP].seg = propa_data.seg_list[0].seg_ref;
	propa_data.tp_data[propa_param.refTP].seg_srs = propa_data.seg_list[0].seg_ref_srs;
	propa_data.tp_data[propa_param.refTP].mesh = propa_data.seg_list[0].mesh_ref;


	// Debug: write out extracted tp images
	if (propa_param.debug)
		{
		for (auto &kv : propa_data.tp_data)
			{
			PropagationTools<TReal>::template WriteImage<TImage3D>(
						kv.second.img,GenerateUnaryTPObjectName("img_", kv.first,
																										propa_param.debug_dir.c_str(), nullptr, ".nii.gz"));

			PropagationTools<TReal>::template WriteImage<TImage3D>(
						kv.second.img_srs, GenerateUnaryTPObjectName("img_", kv.first,
																												 propa_param.debug_dir.c_str(), "_srs", ".nii.gz"));
			}

		PropagationTools<TReal>::template WriteImage<TLabelImage3D>(
					propa_data.tp_data[propa_param.refTP].seg_srs,
				GenerateUnaryTPObjectName("mask_", propa_param.refTP, propa_param.debug_dir.c_str(), "_srs", ".nii.gz"));
		}
}

template<typename TReal>
void
PropagationAPI<TReal>
::CreateTimePointLists(const std::vector<unsigned int> &target, std::vector<unsigned int> &forward,
											 std::vector<unsigned int> &backward, unsigned int refTP)
{
	forward.push_back(refTP);
	backward.push_back(refTP);

	for (unsigned int tp : target)
		{
		if (tp < refTP)
			backward.push_back(tp);
		else if (tp == refTP)
			continue; // ignore reference tp in the target list
		else
			forward.push_back(tp);
		}

	std::sort(forward.begin(), forward.end());
	std::sort(backward.rbegin(), backward.rend()); // sort backward reversely
}

template <typename TReal>
int
PropagationAPI<TReal>
::RunUnidirectionalPropagation(PropagationData<TReal> &propa_data, const GreedyParameters &greedy_param,
																				const std::vector<unsigned int> &tp_list)
{
	const GreedyPropagationParameters &propa_param = greedy_param.propagation_param;
	if (propa_param.debug)
		{
		std::cout << "-- [Propagation] Unidirectional Propagation for tp_list: ";
		for (auto tp : tp_list)
			std::cout << " " << tp;
		std::cout << std::endl;
		}

	RunDownSampledPropagation(propa_data, greedy_param, tp_list); // Generate affine matrices and masks
	GenerateFullResolutionMasks(propa_data, greedy_param, tp_list); // Reslice downsampled masks to full-res
	GenerateReferenceSpace(propa_data, greedy_param, tp_list); // Generate reference space for faster run

	// Run reg between ref and target tp and warp reference segmentation to target segmentation
	for (size_t crnt = 1; crnt < tp_list.size(); ++crnt) // this can be parallelized
		{
		RunFullResolutionPropagation(propa_data, greedy_param, tp_list[crnt]);
		}

	return EXIT_SUCCESS;
}

template <typename TReal>
int
PropagationAPI<TReal>
::RunFullResolutionPropagation(PropagationData<TReal> &propa_data, const GreedyParameters &greedy_param,
															 const unsigned int target_tp)
{
	const GreedyPropagationParameters &propa_param = greedy_param.propagation_param;
	if (propa_param.debug)
		{
		std::cout << "-- [Propagation] Running Full Resolution Propagation from "
							<< "reference tp: " << propa_param.refTP << " to target tp: " << target_tp << std::endl;
		}

	// Run Deformable Reg from target to ref
	RunPropagationDeformable(propa_data, greedy_param, target_tp, propa_param.refTP, true);

	// Warp ref segmentation to target
	RunPropagationReslice(propa_data, greedy_param, propa_param.refTP, target_tp, true);

	// Write out reliced segmentation
	TimePointData<TReal> &target_data = propa_data.tp_data[target_tp];
	std::string fnseg = GenerateUnaryTPObjectName("seg_", target_tp, propa_data.seg_list[0].outdir.c_str(),
																								"_resliced", ".nii.gz");
	if (propa_param.debug)
		{
		std::cout << "-- [Propagation] Write out resliced segmentation for tp: " << target_tp << std::endl;
		std::cout << "---- File path: " << fnseg << std::endl;
		}

	PropagationTools<TReal>::template WriteImage<TLabelImage3D>(target_data.seg, fnseg);

	return EXIT_SUCCESS;
}

template <typename TReal>
void
PropagationAPI<TReal>
::GenerateReferenceSpace(PropagationData<TReal> &propa_data, const GreedyParameters &greedy_param,
												 const std::vector<unsigned int> &tp_list)
{
}

template <typename TReal>
void
PropagationAPI<TReal>
::GenerateFullResolutionMasks(PropagationData<TReal> &propa_data, const GreedyParameters &greedy_param,
															const std::vector<unsigned int> &tp_list)
{
	const GreedyPropagationParameters &propa_param = greedy_param.propagation_param;
	if (propa_param.debug)
		{
		std::cout << "-- [Propagation] Generating Full Resolution Masks " << std::endl;
		}

	for (size_t i = 0; i < tp_list.size(); ++i)
		{
		const unsigned int tp = tp_list[i];
		TimePointData<TReal> &tp_data = propa_data.tp_data[tp];
		tp_data.full_res_mask = PropagationTools<TReal>
				::ResliceLabelImageWithIdentityMatrix(tp_data.img, tp_data.seg_srs);
		std::string fnmask = GenerateUnaryTPObjectName("mask_", tp);
		if (propa_param.debug)
			{
			fnmask = GenerateUnaryTPObjectName("mask_", tp, propa_param.debug_dir.c_str(), nullptr, ".nii.gz");
			PropagationTools<TReal>::template WriteImage<TLabelImage3D>(tp_data.full_res_mask, fnmask);
			}
		tp_data.full_res_mask->SetObjectName(fnmask);
		}
}


template<typename TReal>
int
PropagationAPI<TReal>
::RunDownSampledPropagation(PropagationData<TReal> &propa_data, const GreedyParameters &greedy_param,
														const std::vector<unsigned int> &tp_list)
{
	const GreedyPropagationParameters &propa_param = greedy_param.propagation_param;
	if (propa_param.debug)
		{
		std::cout << "-- [Propagation] Down Sampled Propagation started " << std::endl;
		}

	for (size_t i = 1; i < tp_list.size(); ++i)
		{
		unsigned int c = tp_list[i], p = tp_list[i - 1]; // current tp and previous tp
		RunPropagationAffine(propa_data, greedy_param, p, c); // affine reg current to prev
		RunPropagationDeformable(propa_data, greedy_param, p, c, false); // deformable reg current to prev
		BuildTransformChainForReslice(propa_data, greedy_param, p, c); // build transformation chain for current tp
		RunPropagationReslice(propa_data, greedy_param, propa_param.refTP, c, false); // warp ref mask to current
		}

	return EXIT_SUCCESS;
}

template<typename TReal>
int
PropagationAPI<TReal>
::RunPropagationAffine(PropagationData<TReal> &propa_data, const GreedyParameters &glparam,
											 unsigned int tp_fix, unsigned int tp_mov)
{
	if (glparam.propagation_param.debug)
		{
		std::cout << "-- [Propagation] Running Propagation Affine Started tp_fix=" << tp_fix
							<< "; tp_mov=" << tp_mov << std::endl;
		}

	TimePointData<TReal> &df = propa_data.tp_data[tp_fix], &dm = propa_data.tp_data[tp_mov];

	// Create a new GreedyAPI for affine run and configure
	GreedyApproach<3u, TReal> *GreedyAPI = new GreedyApproach<3u, TReal>();

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
	param.CopyGeneralSettings(glparam); // copy general settings from user input
	param.CopyAffineSettings(glparam); // copy affine settings from user input
	// Override global default settings with propagation specific setting
	param.affine_init_mode = AffineInitMode::RAS_IDENTITY;
	param.affine_dof = GreedyParameters::DOF_RIGID;

	// Check smoothing parameters. If greedy default detected, change to propagation default.
	const SmoothingParameters prop_default_pre = { 3.0, true }, prop_default_post = { 1.5, true };
	param.sigma_pre = (glparam.sigma_pre == GreedyParameters::default_sigma_pre) ? prop_default_pre : glparam.sigma_pre;
	param.sigma_post = (glparam.sigma_post == GreedyParameters::default_sigma_post) ? prop_default_post : glparam.sigma_post;

	// Add the input group to the parameters
	param.input_groups.clear();
	param.input_groups.push_back(ig);

	// Configure output
	bool force_write = false;
	param.output = GenerateBinaryTPObjectName("affine_", tp_mov, tp_fix);

	if (glparam.propagation_param.debug)
		{
		force_write = true;
		param.output = GenerateBinaryTPObjectName("affine_", tp_mov, tp_fix,
																							glparam.propagation_param.debug_dir.c_str(), ".mat");
		std::cout << "-- [Propagation] Affine Command: " << param.GenerateCommandLine() << std::endl;
		}

	dm.affine_to_prev->SetObjectName(param.output);
	GreedyAPI->AddCachedOutputObject(param.output, dm.affine_to_prev, force_write);
	int ret = GreedyAPI->RunAffine(param);

	delete GreedyAPI;
	return ret;
}

template<typename TReal>
int
PropagationAPI<TReal>
::RunPropagationDeformable(PropagationData<TReal> &propa_data, const GreedyParameters &glparam,
																		unsigned int tp_fix, unsigned int tp_mov, bool isFullRes)
{
	if (glparam.propagation_param.debug)
		{
		std::cout << "-- [Propagation] Deformable Run: tp_fix=" << tp_fix
							<< "; tp_mov = " << tp_mov << "; isFullRes=" << isFullRes << std::endl;
		}

	// Get relevant tp data
	TimePointData<TReal> &tpdata_fix = propa_data.tp_data[tp_fix];
	TimePointData<TReal> &tpdata_mov = propa_data.tp_data[tp_mov];

	// Set greedy parameters
	GreedyApproach<3u, TReal> *GreedyAPI = new GreedyApproach<3u, TReal>();
	GreedyParameters param;
	param.mode = GreedyParameters::GREEDY;
	param.CopyDeformableSettings(glparam);
	param.CopyGeneralSettings(glparam);

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
	param.sigma_pre = (glparam.sigma_pre == GreedyParameters::default_sigma_pre) ? prop_default_pre : glparam.sigma_pre;
	param.sigma_post = (glparam.sigma_post == GreedyParameters::default_sigma_post) ? prop_default_post : glparam.sigma_post;

	// Configure output
	bool force_write = false; // Write out images for debugging
	const char *suffix = isFullRes ? "" : "_srs";
	param.output = GenerateBinaryTPObjectName("warp_", tp_mov, tp_fix, nullptr, suffix);
	param.inverse_warp = GenerateBinaryTPObjectName("warp_", tp_fix, tp_mov, nullptr, suffix);

	if (glparam.propagation_param.debug)
		{
		force_write = true;
		param.output = GenerateBinaryTPObjectName("warp_", tp_mov, tp_fix,
				glparam.propagation_param.debug_dir.c_str(), suffix, ".nii.gz");
		param.inverse_warp = GenerateBinaryTPObjectName("warp_", tp_fix, tp_mov,
				glparam.propagation_param.debug_dir.c_str(), suffix, ".nii.gz");
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

	if (glparam.propagation_param.debug)
		std::cout << "-- [Propagation] Deformable Command: " <<  param.GenerateCommandLine() << std::endl;

	int ret = GreedyAPI->RunDeformable(param);

	delete GreedyAPI;
	return ret;
}

template<typename TReal>
int
PropagationAPI<TReal>
::RunPropagationReslice(PropagationData<TReal> &propa_data, const GreedyParameters &glparam,
																 unsigned int tp_in, unsigned int tp_out, bool isFullRes)
{
	if (glparam.propagation_param.debug)
		{
		std::cout << "-- [Propagation] Reslice Run. tp_in=" << tp_in << "; tp_out="
							<< tp_out << "; isFullRes=" << isFullRes << std::endl;
		}

	TimePointData<TReal> &tpdata_in = propa_data.tp_data[tp_in];
	TimePointData<TReal> &tpdata_out = propa_data.tp_data[tp_out];

	// API and parameter configuration
	GreedyApproach<3u, TReal> *GreedyAPI = new GreedyApproach<3u, TReal>();
	GreedyParameters param;
	param.mode = GreedyParameters::RESLICE;
	param.CopyGeneralSettings(glparam);
	param.CopyReslicingSettings(glparam);

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
		imgout_name = GenerateUnaryTPObjectName("seg_", tp_out, propa_data.seg_list[0].outdir.c_str(), "_resliced", ".nii.gz");
		}
	else if (param.propagation_param.debug)
		{
		force_write = true;
		imgout_name = GenerateUnaryTPObjectName("mask_", tp_out, param.propagation_param.debug_dir.c_str(), "_srs", ".nii.gz");
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
	ResliceSpec rspec(imgin_name, imgout_name, glparam.propagation_param.reslice_spec);
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

	delete GreedyAPI;
	return ret;
}

template<typename TReal>
void
PropagationAPI<TReal>
::BuildTransformChainForReslice(PropagationData<TReal> &propa_data, const GreedyParameters &glparam,
																unsigned int tp_prev, unsigned int tp_crnt)
{
	if (glparam.propagation_param.debug)
		{
		std::cout << "-- [Propagation] Building reslicing trans chain for tp: " << tp_crnt << std::endl;
		}

	TimePointData<TReal> &tpdata_crnt = propa_data.tp_data[tp_crnt];
	TimePointData<TReal> &tpdata_prev = propa_data.tp_data[tp_prev];

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
