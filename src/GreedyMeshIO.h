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

vtkSmartPointer<vtkPointSet> ReadMesh(const char *fname);
void WriteMesh(vtkPointSet *mesh, const char *fname);

#endif // GREEDYMESHIO_H
