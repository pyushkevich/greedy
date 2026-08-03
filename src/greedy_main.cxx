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
