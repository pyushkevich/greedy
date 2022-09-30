#ifndef PROPAGATIONAPI_H
#define PROPAGATIONAPI_H

#include "lddmm_data.h"
#include "GreedyParameters.h"
#include "PropagationParameters.h"

#include <memory>
#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <itkMatrixOffsetTransformBase.h>

namespace propagation
{
template<typename TReal>
class PropagationData;

template<typename TReal>
class PropagationInput;

template<typename TReal>
class PropagationOutput;

template<typename TReal>
class PropagationTools;

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
  using TPropagationMesh = vtkPolyData;
  using TPropagationMeshPointer = vtkSmartPointer<vtkPolyData>;
  using PTools = PropagationTools<TReal>;

	enum ResampleInterpolationMode { Linear=0, NearestNeighbor };

  PropagationAPI() = delete;

  /** Specialized constructor for api run */
  PropagationAPI(const std::shared_ptr<PropagationInput<TReal>> input);

	~PropagationAPI();
	PropagationAPI(const PropagationAPI &other) = delete;
	PropagationAPI &operator=(const PropagationAPI &other) = delete;
  /** Start the execution of the propagation pipeline */
  int Run();

private:
  void ValidateInputData();
  void PrepareTimePointData();
  void ValidateInputOrientation();
  void CreateReferenceMask();
  void CreateTimePointLists();
  void Generate4DSegmentation();

  void RunUnidirectionalPropagation(const std::vector<unsigned int> &tp_list);
  void RunDownSampledPropagation(const std::vector<unsigned int> &tp_list);
  void GenerateFullResolutionMasks(const std::vector<unsigned int> &tp_list);
  void GenerateReferenceSpace(const std::vector<unsigned int> &tp_list);
  void RunFullResolutionPropagation(const unsigned int target_tp);

  void RunPropagationAffine(unsigned int tp_fix, unsigned int tp_mov);
  void RunPropagationDeformable(unsigned int tp_fix, unsigned int tp_mov, bool isFullRes);
  void RunPropagationReslice(unsigned int tp_in, unsigned int tp_out, bool isFullRes);
  void RunPropagationMeshReslice(unsigned int tp_in, unsigned int tp_out);
  void BuildTransformChainForReslice(unsigned int tp_prev, unsigned int tp_crnt);

  static inline std::string GenerateUnaryTPObjectName(const char *base, unsigned int tp,
      const char *debug_dir = nullptr, const char *suffix = nullptr, const char *file_ext = nullptr);

	static inline std::string GenerateBinaryTPObjectName(const char *base, unsigned int tp1, unsigned int tp2,
      const char *debug_dir = nullptr, const char *suffix = nullptr, const char *file_ext = nullptr);

  static inline std::string GenerateUnaryTPFileName(const char *pattern, unsigned int tp,
      const char *output_dir = nullptr, const char *file_ext = nullptr);

  std::shared_ptr<PropagationData<TReal>> m_Data;
  GreedyParameters m_GParam;
  PropagationParameters m_PParam;
  std::vector<unsigned int> m_ForwardTPs;
  std::vector<unsigned int> m_BackwardTPs;
};

}
#endif // PROPAGATIONAPI_H
