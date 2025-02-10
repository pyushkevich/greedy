#include "MultiChunkGreedy.h"
#include "CommandLineHelper.h"

int main(int argc, char *argv[])
{
  if(argc < 2)
    return multichunk_greedy_usage();

  // Parse the command line
  CommandLineHelper cl(argc, argv);
  auto [chunk_greedy_param, greedy_param] = greedy_multi_chunk_parse_parameters(cl, true);

  // Check the dimension
  if(greedy_param.dim == 2)
    return run_multichunk_greedy<2>(chunk_greedy_param, greedy_param);
  else if(greedy_param.dim == 3)
    return run_multichunk_greedy<3>(chunk_greedy_param, greedy_param);
}
