#ifndef POINTSETGEODESICTOWARP_H
#define POINTSETGEODESICTOWARP_H

#include <string>
#include <list>
#include "PointSetHamiltonianSystem.h"
#include "CommandLineHelper.h"
#include "lddmm_data.h"

class vtkPolyData;

struct WarpGenerationParameters
{
  typedef std::pair<std::string, std::string> MeshPair;

  std::string fnReference, fnMesh, fnOutWarp, fnMask;
  double sigma = 0.0;
  unsigned int dim = 3;
  unsigned int N = 0;
  bool use_ralston_method = false;
  unsigned int anim_freq = 0;
  unsigned int n_threads = 0;
  bool brute = false;
  bool use_float = false;

  std::list<MeshPair> fnWarpMeshes;
};


template <class TPixel, unsigned int VDim>
class PointSetGeodesicToWarp
{
public:
  typedef PointSetHamiltonianSystem<double, VDim> HSystem;
  typedef typename HSystem::Vector Vector;
  typedef typename HSystem::Matrix Matrix;

  typedef LDDMMData<TPixel, VDim> LDDMM;
  typedef typename LDDMM::ImageType ImageType;
  typedef typename LDDMM::VectorImageType VectorImageType;
  typedef typename LDDMM::ImagePointer ImagePointer;
  typedef typename LDDMM::VectorImagePointer VectorImagePointer;

  typedef typename ImageType::PointType PointType;
  typedef itk::ContinuousIndex<TPixel,VDim> ContIndexType;

  static int run(const WarpGenerationParameters &param);

  static VectorImagePointer brute_force_method(
      const WarpGenerationParameters &param,
      HSystem &hsys, int tdir, int tStart, int tEnd);

private:

  static void UpdateAndWriteMesh(
      vtkPolyData *mesh, const Matrix &x, const Matrix &v, const Matrix &x0,
      const std::string filePattern,
      int k);
};

int lmtowarp_usage(bool print_template_param);

// Parse command line and return parameter structure
WarpGenerationParameters lmtowarp_parse_commandline(CommandLineHelper &cl, bool parse_template_params = true);

#endif // POINTSETGEODESICTOWARP_H
