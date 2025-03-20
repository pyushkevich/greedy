#ifndef __PointSetGeodesicShooting_h_
#define __PointSetGeodesicShooting_h_

#include "PointSetHamiltonianSystem.h"
#include "CommandLineHelper.h"
#include <random>

struct ShootingParameters
{
  enum Algorithm { Allassonniere, GradDescent, QuasiAllassonniere };
  enum DataAttachment { Euclidean, Current, Varifold };
  std::string fnTemplate, fnTarget, fnControlMesh;
  std::string fnOutput;
  std::string fnOutputPaths;
  std::string arrInitialMomentum;
  std::string arrAttachmentLabelPosteriors;
  double sigma = 0.0;
  double currents_sigma = 0.0;
  double lambda = 1.0;
  double gamma = 1.0;
  unsigned int dim = 3;
  unsigned int N = 100;
  bool use_ralston_method = false;
  unsigned int iter_grad = 20, iter_newton = 20;
  Algorithm alg = GradDescent;
  DataAttachment attach = Euclidean;
  bool use_float = false;
  unsigned int n_threads = 0;
  unsigned int n_deriv_check = 0;
  bool test_currents_attachment = false;
  bool do_similarity_matching = false;
  int random_seed = 0;

  // Weight for the mesh Jacobian penalty term
  double w_jacobian = 0.0;

  // For constrained optimization - just exprimental
  double constrained_mu_init = 0.0, constrained_mu_mult = 0.0;
};


template <class TFloat, unsigned int VDim>
class PointSetShootingProblem
{
public:
  typedef PointSetHamiltonianSystem<TFloat, VDim> HSystem;
  typedef typename HSystem::Vector Vector;
  typedef typename HSystem::Matrix Matrix;
  typedef vnl_matrix<int> Triangulation;

  // Minimize using the transversality principle
  static void minimize_Allassonniere(const ShootingParameters &param,
                                     const Matrix &q0, const Matrix &qT, Matrix &p0);

  static void minimize_QuasiAllassonniere(const ShootingParameters &param,
                                          const Matrix &q0, const Matrix &qT, Matrix &p0);

  // Minimize using gradient descent
  static void minimize_gradient(
      const ShootingParameters &param,
      const Matrix &q0, const Matrix &qT, Matrix &p0,
      const Triangulation &tri_template, const Triangulation &tri_target,
      const Matrix &lab_template, const Matrix &lab_target);

  static int similarity_matching(
      const ShootingParameters &param,
      const Matrix &q0, const Matrix &qT, Matrix &q0_sim, Matrix &qT_sim,
      const Triangulation &tri_template, const Triangulation &tri_target,
      const Matrix &lab_template, const Matrix &lab_target);

  static int minimize(const ShootingParameters &param);

private:
  static int TestCurrentsAttachmentTerm(
      const ShootingParameters &param,
      Matrix &q0, Matrix &qT,
      vnl_matrix<int> &tri_template, vnl_matrix<int> &tri_target,
      const Matrix &lab_template, const Matrix &lab_target);

  static std::mt19937 m_Random;
};

// Usage
int lmshoot_usage(bool print_template_params);

// Parse command line and return parameter structure
ShootingParameters lmshoot_parse_commandline(CommandLineHelper &cl, bool parse_template_params);


#endif
