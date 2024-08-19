#include "PointSetHamiltonianSystem.h"
#include <iostream>
#include "util/ReadWriteVTK.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "vtkFloatArray.h"
#include "vtkSmartPointer.h"

#include <cstdarg>

using namespace std;

int usage()
{
  cout << "lmflow: Apply geodesic shooting to a mesh" << endl;
  cout << "Usage:" << endl;
  cout << "  lmflow [options]" << endl;
  cout << "Required options:" << endl;
  cout << "  -m mesh.vtk        : Mesh with the InitialMomentum array defined" << endl;
  cout << "  -s sigma           : kernel standard deviation" << endl;
  cout << "  -o out.vtk         : output mesh" << endl;
  cout << "Additional options:" << endl;
  cout << "  -d dim                     : problem dimension (3)" << endl;
  cout << "  -n N                       : number of time steps (100)" << endl;

  return -1;
}

void check(bool condition, const char *format,...)
{
  if(!condition)
    {
    char buffer[256];
    va_list args;
    va_start (args, format);
    vsprintf (buffer,format, args);
    va_end (args);

    cerr << buffer << endl;
    exit(-1);
    }
}


struct FlowParameters
{
  string fnMesh, fnOutMesh;
  double sigma;
  unsigned int dim;
  unsigned int N;

  FlowParameters():
    N(100), dim(3), sigma(0.0) {}
};

template <class TPixel, unsigned int VDim>
class PointSetFlow
{
public:
  typedef PointSetHamiltonianSystem<double, VDim> HSystem;
  typedef typename HSystem::Vector Vector;
  typedef typename HSystem::Matrix Matrix;

  static int run(const FlowParameters &param);
};

template <class TPixel, unsigned int VDim>
int
PointSetFlow<TPixel, VDim>
::run(const FlowParameters &param)
{
  // Read the VTK mesh containing the points
  vtkPolyData *mesh = ReadVTKData(param.fnMesh);
  if(!mesh)
    {
    cerr << "Failed to read mesh from " << param.fnMesh << endl;
    return -1;
    }

  // Read the momentum field
  vtkDataArray *arr_p0 = mesh->GetPointData()->GetArray("InitialMomentum");
  if(!arr_p0 || arr_p0->GetNumberOfComponents() != VDim)
    {
    cerr << "Failed to read initial momentum from " << param.fnMesh << endl;
    return -1;
    }

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
  HSystem hsys(q0, param.sigma, param.N, 0, 0);

  // Flow without gradients - we have streamlines
  if(param.N > 0)
    {
    hsys.FlowHamiltonian(p0, q1, p1);
    }
  else
    {
    hsys.ComputeHamiltonianJet(q0, p0, false);
    q1 = q0;
    p1 = p0;
    }
 
  // Store the velocity
  vtkSmartPointer<vtkFloatArray> arr_p1 = vtkFloatArray::New();
  arr_p1->SetNumberOfComponents(VDim); 
  arr_p1->SetNumberOfTuples(arr_p0->GetNumberOfTuples());
  arr_p1->SetName("FinalMomentum");
  for(int a = 0; a < VDim; a++)
    arr_p1->FillComponent(a, NAN);
  mesh->GetPointData()->AddArray(arr_p1);

  vtkSmartPointer<vtkFloatArray> arr_v = vtkFloatArray::New();
  arr_v->SetNumberOfComponents(VDim); 
  arr_v->SetNumberOfTuples(arr_p0->GetNumberOfTuples());
  arr_v->SetName("Velocity");
  for(int a = 0; a < VDim; a++)
    arr_v->FillComponent(a, NAN);
  mesh->GetPointData()->AddArray(arr_v);

  for(unsigned int i = 0; i < k; i++)
    {
    double x[VDim];
    for(unsigned int a = 0; a < VDim; a++)
      {
      x[a] = q1(i,a);
      arr_v->SetComponent(index[i], a, hsys.GetHp(a)[i]);
      arr_p1->SetComponent(index[i], a, p1(i,a));
      }
    mesh->GetPoints()->SetPoint(index[i], x);
    }

  WriteVTKData(mesh,param.fnOutMesh);

  return 0;
}



int main(int argc, char *argv[])
{
  FlowParameters param;

  if(argc < 2)
    return usage();

  // Process parameters
  for(int i = 1; i < argc; i++)
    {
    string arg = argv[i];
    if(arg == "-m" && i < argc-1)
      {
      param.fnMesh = argv[++i];
      }
    else if(arg == "-o" && i < argc-1)
      {
      param.fnOutMesh = argv[++i];
      }
    else if(arg == "-s" && i < argc-1)
      {
      param.sigma = atof(argv[++i]);
      }
    else if(arg == "-d" && i < argc-1)
      {
      param.dim = (unsigned int) atoi(argv[++i]);
      }
    else if(arg == "-n" && i < argc-1)
      {
      param.N = (unsigned int) atoi(argv[++i]);
      }
    else if(arg == "-h")
      {
      return usage();
      }
    else
      {
      cerr << "Unknown option " << arg << endl;
      return -1;
      }
    }

  check(param.sigma > 0, "Missing or negative sigma parameter");
  check(param.N < 10000, "Incorrect N parameter");
  check(param.dim >= 2 && param.dim <= 3, "Incorrect N parameter");
  check(param.fnMesh.length(), "Missing target filename");
  check(param.fnOutMesh.length(), "Missing output filename");

  // Specialize by dimension
  if(param.dim == 2)
    return PointSetFlow<float,2>::run(param);
  else
    return PointSetFlow<float,3>::run(param);

  return 0;
}
