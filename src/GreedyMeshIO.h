#ifndef GREEDYMESHIO_H
#define GREEDYMESHIO_H

#include <vtkSmartPointer.h>

class vtkPolyDataReader;
class vtkPolyDataWriter;
class vtkPLYReader;
class vtkPLYWriter;
class vtkSTLReader;
class vtkSTLWriter;
class vtkBYUReader;
class vtkBYUWriter;
class vtkOBJReader;
class vtkOBJWriter;
class vtkPointSet;
class vtkPolyData;

vtkSmartPointer<vtkPointSet> ReadMesh(const char *fname);
vtkSmartPointer<vtkPolyData> ReadVTKPolyData(const char *fname);
void WriteMesh(vtkPointSet *mesh, const char *fname);

void WriteJacobianMesh(vtkPointSet *fixed_mesh, vtkPointSet *moving_mesh, const char *fname);

vtkSmartPointer<vtkPointSet> DeepCopyMesh(vtkPointSet *mesh);


#endif // GREEDYMESHIO_H
