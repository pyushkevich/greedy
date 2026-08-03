/*=========================================================================

  Program:   GreedyReg fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  GreedyReg initial development was funded by the NIH grant R01 EB017255.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

=========================================================================*/
#ifndef LINEARTRANSFORMTOWARPFILTER_TXX
#define LINEARTRANSFORMTOWARPFILTER_TXX

#include "LinearTransformToWarpFilter.h"
#include "ImageRegionConstIteratorWithIndexOverride.h"


template <class TInputImage, class TDeformationField, class TTransform>
void
LinearTransformToWarpFilter<TInputImage,TDeformationField,TTransform>
::DynamicThreadedGenerateData(const OutputImageRegionType& outputRegionForThread)
{



  // Get a pointer to the output deformation field
  DeformationVectorType *b_phi = this->GetOutput()->GetBufferPointer();

  // Affine transform matrix and vector
  vnl_matrix_fixed<double, ImageDimension, ImageDimension> M =
      this->GetTransform()->GetMatrix().GetVnlMatrix();
  vnl_vector_fixed<double, ImageDimension> off =
      this->GetTransform()->GetOffset().GetVnlVector();

  // Create an iterator over the deformation field
  typedef itk::ImageLinearIteratorWithIndex<DeformationFieldType> IterBase;
  typedef IteratorExtender<IterBase> IterType;

  // Loop over the lines in the image
  for(IterType it(this->GetOutput(), outputRegionForThread); !it.IsAtEnd(); it.NextLine())
    {
    // Get the index at the current location. For the rest of the line, the index will
    // increment by one
    IndexType idx = it.GetIndex();

    // Displacement vector for the first position in the line and a delta corresponding to a
    // step along the line
    DeformationVectorType disp, delta_disp;

    // Map to a position at which to interpolate
    for(int i = 0; i < ImageDimension; i++)
      {
      disp[i] = off[i] - idx[i];
      delta_disp[i] = M(i, 0);
      for(int j = 0; j < ImageDimension; j++)
        disp[i] += M(i,j) * idx[j];
      }
    delta_disp[0] -= 1.0;

    // Pointer to the start and end of the line
    DeformationVectorType *p_phi = const_cast<DeformationVectorType *>(it.GetPosition());
    DeformationVectorType *p_phi_end = p_phi + outputRegionForThread.GetSize(0);

    // Run a loop filling out the displacement field
    for( ; p_phi < p_phi_end; ++p_phi, disp += delta_disp)
      {
      *p_phi = disp;
      }

    /*
    for( ; p_phi < p_phi_end; ++p_phi, ++idx[0])
      {
      for(int i = 0; i < ImageDimension; i++)
        {
        (*p_phi)[i] = off[i] - idx[i];
        for(int j = 0; j < ImageDimension; j++)
          (*p_phi)[i] += M(i,j) * idx[j];
        }
      }

*/
    }
}


#endif // LINEARTRANSFORMTOWARPFILTER_TXX

