#ifndef PROPAGATIONAPI_H
#define PROPAGATIONAPI_H

#include "GreedyParameters.h"
#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <vtkSmartPointer.h>
#include <itkMatrixOffsetTransformBase.h>
#include <vtkPolyData.h>
#include "lddmm_data.h"

namespace propagation
{

template<typename TReal>
class PropagationData;

template<typename TReal>
class PropagationAPI
{
public:
	using TImage4D = itk::Image<TReal, 4>;
	using TImage3D = itk::Image<TReal, 3>;
	using TLabelImage4D = itk::Image<short, 4>;
	using TLabelImage3D = itk::Image<short, 3>;
	using TLDDMM3D = LDDMMData<TReal, 3>;
	using TVectorImage3D = typename TLDDMM3D::VectorImageType;
	using TCompositeImage3D = typename TLDDMM3D::CompositeImageType;
	using TTransform = itk::MatrixOffsetTransformBase<TReal, 3, 3>;
	using TMeshPointer = vtkSmartPointer<vtkPolyData>;

	enum ResampleInterpolationMode { Linear=0, NearestNeighbor };

	PropagationAPI();
	~PropagationAPI();
	PropagationAPI(const PropagationAPI &other) = delete;
	PropagationAPI &operator=(const PropagationAPI &other) = delete;

	static int Run(const GreedyParameters &param);

private:
	static void PrepareTimePointData(PropagationData<TReal> &propa_data, const GreedyPropagationParameters &propa_param);
	static void CreateTimePointLists(const std::vector<unsigned int> &target_list, std::vector<unsigned int> &forward_list,
														std::vector<unsigned int> &backward_list, unsigned int refTP);
	static int RunUnidirectionalPropagation(PropagationData<TReal> &propa_data, const GreedyParameters &greedy_param,
																					const std::vector<unsigned int> &tp_list);
	static int RunDownSampledPropagation(PropagationData<TReal> &propa_data, const GreedyParameters &greedy_param,
																			 const std::vector<unsigned int> &tp_list);
	static void GenerateFullResolutionMasks(PropagationData<TReal> &propa_data, const GreedyParameters &greedy_param,
																				 const std::vector<unsigned int> &tp_list);

	static void GenerateReferenceSpace(PropagationData<TReal> &propa_data, const GreedyParameters &greedy_param,
																		 const std::vector<unsigned int> &tp_list);

	static int RunFullResolutionPropagation(PropagationData<TReal> &propa_data, const GreedyParameters &greedy_param,
																					 const unsigned int target_tp);

	static int RunPropagationAffine(PropagationData<TReal> &propa_data, const GreedyParameters &glparam,
														unsigned int tp_fix, unsigned int tp_mov);
	static int RunPropagationDeformable(PropagationData<TReal> &propa_data, const GreedyParameters &glparam,
																			unsigned int tp_fix, unsigned int tp_mov, bool isFullRes);
	static int RunPropagationReslice(PropagationData<TReal> &propa_data, const GreedyParameters &glparam,
																	 unsigned int tp_in, unsigned int tp_out, bool isFullRes);

	static void BuildTransformChainForReslice(PropagationData<TReal> &propa_data, const GreedyParameters &propa_param,
																						unsigned int tp_prev, unsigned int tp_crnt);

	static inline std::string GenerateUnaryTPObjectName(const char *base, unsigned int tp,
																											const char *debug_dir = nullptr, const char *suffix = nullptr,
																											const char *file_ext = nullptr);

	static inline std::string GenerateBinaryTPObjectName(const char *base, unsigned int tp1, unsigned int tp2,
																											 const char *debug_dir = nullptr, const char *suffix = nullptr,
																											 const char *file_ex = nullptr);
};


}
#endif // PROPAGATIONAPI_H
