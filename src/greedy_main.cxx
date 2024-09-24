/*=========================================================================

  Program:   ALFABIS fast medical image registration programs
  Language:  C++
  Website:   github.com/pyushkevich/greedy
  Copyright (c) Paul Yushkevich, University of Pennsylvania. All rights reserved.

  This program is part of ALFABIS: Adaptive Large-Scale Framework for
  Automatic Biomedical Image Segmentation.

  ALFABIS development is funded by the NIH grant R01 EB017255.

  ALFABIS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ALFABIS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ALFABIS.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================*/

#include "GreedyAPI.h"
#include "CommandLineHelper.h"

#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cerrno>

#include "lddmm_common.h"
#include "lddmm_data.h"

#include <itkImageFileReader.h>
#include <itkAffineTransform.h>
#include <itkTransformFactory.h>
#include <itkTimeProbe.h>

#include "MultiImageRegistrationHelper.h"
#include "FastWarpCompositeImageFilter.h"
#include <vnl/vnl_cost_function.h>
#include <vnl/vnl_random.h>
#include <vnl/algo/vnl_powell.h>
#include <vnl/algo/vnl_svd.h>
#include <vnl/vnl_trace.h>

extern const char *GreedyVersionInfo;


template <unsigned int VDim, typename TReal>
class GreedyRunner
{
public:
  static int Run(GreedyParameters &param)
  {
    // Use the threads parameter
    GreedyApproach<VDim, TReal> greedy;   
    return greedy.Run(param);
  }
};



int main(int argc, char *argv[])
{
  GreedyParameters param;

  if(argc < 2)
    return greedy_usage();

  try
  {
    // Parse the command line
    CommandLineHelper cl(argc, argv);
    param = greedy_parse_commandline(cl, true);

    // Run the main code
    if(param.flag_float_math)
      {
      switch(param.dim)
        {
        case 2: return GreedyRunner<2, float>::Run(param); break;
        case 3: return GreedyRunner<3, float>::Run(param); break;
        case 4: return GreedyRunner<4, float>::Run(param); break;
        default: throw GreedyException("Wrong number of dimensions requested: %d", param.dim);
        }
      }
    else
      {
      switch(param.dim)
        {
        case 2: return GreedyRunner<2, double>::Run(param); break;
        case 3: return GreedyRunner<3, double>::Run(param); break;
        case 4: return GreedyRunner<4, double>::Run(param); break;
        default: throw GreedyException("Wrong number of dimensions requested: %d", param.dim);
        }
      }
  }
  catch(std::exception &exc)
  {
    std::cerr << "ABORTING PROGRAM DUE TO RUNTIME EXCEPTION -- "
              << exc.what() << std::endl;
    return -1;
  }
}
