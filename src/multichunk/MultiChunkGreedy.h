#ifndef GREEDYMULTICHUNK_H
#define GREEDYMULTICHUNK_H

#include <string>
#include <vector>
#include <GreedyParameters.h>

struct ChunkGreedyParameters
{
  std::string fn_chunk_mask;
  std::string fn_output_pattern, fn_output_inv_pattern, fn_output_root_pattern, fn_output_metric_gradient_pattern;
  std::string fn_init_tran_pattern;
  std::vector<TransformSpec> transforms_pattern;
  std::vector<TransformSpec> moving_pre_transforms_pattern;
  std::vector<int> crop_margin;
  double reg_weight = 0.01;
};

int multichunk_greedy_usage();

std::pair<ChunkGreedyParameters, GreedyParameters>
greedy_multi_chunk_parse_parameters(CommandLineHelper &cl, bool parse_template_params);

template <unsigned int VDim> int run_multichunk_greedy(ChunkGreedyParameters cgp, GreedyParameters gp);

#endif // GREEDYMULTICHUNK_H
