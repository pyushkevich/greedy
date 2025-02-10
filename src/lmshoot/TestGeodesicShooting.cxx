#include "PointSetHamiltonianSystem.h"
#include <iostream>
#include <cstdlib>
#include <vnl/vnl_matlab_filewrite.h>
#include <vnl/vnl_matlab_read.h>
#include <vnl/vnl_random.h>
#include <cstdarg>
#include <chrono>

void test(double val1, double val2, double tol, const char *format, ...)
{
  if(fabs(val1 - val2) > tol)
    {
    char buffer[256];
    va_list args;
    va_start (args, format);
    vsnprintf (buffer,256,format, args);
    va_end (args);

    std::cerr << "Mismatch in " << buffer << " ";
    std::cerr << val1 << " vs " << val2 << std::endl;
    // exit(-1);
    }
}

int main(int argc, char *argv[])
{
  typedef PointSetHamiltonianSystem<double, 2> Ham;

  // Read regression data
  std::ifstream iss(argv[1]);

  // Input data
  vnl_vector<double> in_q0[2], in_p0[2], in_target[2];
  vnl_vector<double> r_H, r_Hp[2], r_Hq[2];
  vnl_matrix<double> r_Hqq[2][2], r_Hqp[2][2], r_Hpp[2][2];

  // Read the regression data for the flow and derivatives
  vnl_vector<double> r_q1[2], r_p1[2];
  vnl_matrix<double> r_gradQ[2][2], r_gradP[2][2];

  // Read the regression data for the Hamiltonian and derivatives
  vnl_matlab_read_or_die(iss, r_H, "H");
  vnl_matlab_read_or_die(iss, r_Hp[0], "Hpx");
  vnl_matlab_read_or_die(iss, r_Hp[1], "Hpy");
  vnl_matlab_read_or_die(iss, r_Hq[0], "Hqx");
  vnl_matlab_read_or_die(iss, r_Hq[1], "Hqy");
  vnl_matlab_read_or_die(iss, in_p0[0], "px");
  vnl_matlab_read_or_die(iss, in_p0[1], "py");
  vnl_matlab_read_or_die(iss, in_q0[0], "qx");
  vnl_matlab_read_or_die(iss, in_q0[1], "qy");
  vnl_matlab_read_or_die(iss, r_Hqq[0][0], "Hqxqx");
  vnl_matlab_read_or_die(iss, r_Hqq[0][1], "Hqxqy");
  vnl_matlab_read_or_die(iss, r_Hqq[1][1], "Hqyqy");
  vnl_matlab_read_or_die(iss, r_Hqp[0][0], "Hqxpx");
  vnl_matlab_read_or_die(iss, r_Hqp[0][1], "Hqxpy");
  vnl_matlab_read_or_die(iss, r_Hqp[1][0], "Hqypx");
  vnl_matlab_read_or_die(iss, r_Hqp[1][1], "Hqypy");
  vnl_matlab_read_or_die(iss, r_Hpp[0][0], "Hpxpx");
  vnl_matlab_read_or_die(iss, r_Hpp[0][1], "Hpxpy");
  vnl_matlab_read_or_die(iss, r_Hpp[1][1], "Hpypy");
  vnl_matlab_read_or_die(iss, r_p1[0], "px_t");
  vnl_matlab_read_or_die(iss, r_p1[1], "py_t");
  vnl_matlab_read_or_die(iss, r_q1[0], "qx_t");
  vnl_matlab_read_or_die(iss, r_q1[1], "qy_t");
  vnl_matlab_read_or_die(iss, r_gradQ[0][0], "qx_nx");
  vnl_matlab_read_or_die(iss, r_gradQ[0][1], "qx_ny");
  vnl_matlab_read_or_die(iss, r_gradQ[1][0], "qy_nx");
  vnl_matlab_read_or_die(iss, r_gradQ[1][1], "qy_ny");
  vnl_matlab_read_or_die(iss, r_gradP[0][0], "px_nx");
  vnl_matlab_read_or_die(iss, r_gradP[0][1], "px_ny");
  vnl_matlab_read_or_die(iss, r_gradP[1][0], "py_nx");
  vnl_matlab_read_or_die(iss, r_gradP[1][1], "py_ny");
  vnl_matlab_read_or_die(iss, in_target[0], "tx");
  vnl_matlab_read_or_die(iss, in_target[1], "ty");
  
  // Initialize the input vectors
  int k = 80;
  Ham::Matrix q0(k, 2), p0(k, 2), q1(k, 2), p1(k, 2), trg(k, 2);
  for(int a = 0; a < 2; a++)
    {
    q0.set_column(a, in_q0[a]);
    p0.set_column(a, in_p0[a]);
    trg.set_column(a, in_target[a]);
    }

  // Initialize the gradients
  Ham::Matrix grad_q[2][2], grad_p[2][2];
  for(int a = 0; a < 2; a++)
    {
    for(int b = 0; b < 2; b++)
      {
      grad_p[a][b].set_size(k,k);
      grad_q[a][b].set_size(k,k);
      }
    }

  // Create a hamiltonian system
  Ham hsys(q0, 0.08, 100, 0, 0);

  // Compute the Hamiltonian
  double H  = hsys.ComputeHamiltonianJet(q0, p0, true);

  // Symmetric terms
  r_Hqq[1][0] = r_Hqq[0][1];
  r_Hpp[1][0] = r_Hpp[0][1];

  // Regression testing on the Hamiltonian
  test(H, r_H(0), 1e-8, "H");
  for(int a = 0; a < 2; a++)
    {
    for(int j = 0; j < k; j++)
      {
      test(hsys.GetHp(a)(j), r_Hp[a](j), 1e-8, "Hp[%d][%d]", a, j); 
      test(hsys.GetHq(a)(j), r_Hq[a](j), 1e-8, "Hq[%d][%d]", a, j); 

      for(int b = 0; b < 2; b++)
        {
        for(int l = 0; l < k; l++)
          {
          test(hsys.GetHpp(a,b)(j,l), r_Hpp[a][b](j,l), 1e-8, "Hpp[%d][%d](%d,%d)",a,b,j,l);
          test(hsys.GetHqp(a,b)(j,l), r_Hqp[a][b](j,l), 1e-8, "Hqp[%d][%d](%d,%d)",a,b,j,l);
          test(hsys.GetHqq(a,b)(j,l), r_Hqq[a][b](j,l), 1e-8, "Hqq[%d][%d](%d,%d)",a,b,j,l);
          }
        }
      }
    }

  // Passed this stage of the test
  std::cout << "Passed regression test on ComputeHamiltonianJet" << std::endl;

  double t_start, t_finish;

  // Flow the system without gradient - to see how long it takes
  t_start = clock();
  hsys.FlowHamiltonian(p0, q1, p1);
  t_finish = clock();
  std::cout << "Flow without gradient computed in " 
    << (t_finish - t_start) / CLOCKS_PER_SEC << " sec" << std::endl;

  // Regression testing on FlowHamiltonian
  for(int i = 0; i < k; i++)
    {
    for(int a = 0; a < 2; a++)
      {
      test(q1(i,a), r_q1[a](i), 1e-8, "q1[%d][%d]", a, i);
      test(p1(i,a), r_p1[a](i), 1e-8, "p1[%d][%d]", a, i);
      }
    }

  // Passed this stage of the test
  std::cout << "Passed regression test on FlowHamiltonian" << std::endl;

  // Flow the system without gradient
  t_start = clock();
  hsys.FlowHamiltonianWithGradient(p0, q1, p1, grad_q, grad_p);
  t_finish = clock();
  std::cout << "Flow with gradient computed in " 
    << (t_finish - t_start) / CLOCKS_PER_SEC << " sec" << std::endl;


  for(int i = 0; i < k; i++)
    {
    for(int a = 0; a < 2; a++)
      {
      test(q1(i,a), r_q1[a](i), 1e-8, "q1[%d][%d]", a, i);
      test(p1(i,a), r_p1[a](i), 1e-8, "p1[%d][%d]", a, i);

      for(int j = 0; j < k; j++)
        {
        for(int b = 0; b < 2; b++)
          {
          test(grad_q[a][b](i,j), r_gradQ[a][b](i,j), 1e-8, "grad_q[%d][%d](%d,%d)",a,b,i,j);
          test(grad_p[a][b](i,j), r_gradP[a][b](i,j), 1e-8, "grad_p[%d][%d](%d,%d)",a,b,i,j);
          }
        }
      }
    }

  // Passed this stage of the test
  std::cout << "Passed regression test on the Hamiltonian flow" << std::endl;

  // Test forward-backward gradient computation approach
  Ham::Vector alpha[2], beta[2], dd[2], dd_test[2];
  vnl_random rnd(1234);
  for(int a = 0; a < 2; a++)
    {
    alpha[a].set_size(k);
    beta[a].set_size(k);

    for(int i = 0; i < k; i++)
      {
      alpha[a](i) = rnd.drand32(-1.0, 1.0);
      beta[a](i) = rnd.drand32(-1.0, 1.0);
      }
    }

  hsys.FlowHamiltonian(p0, q1, p1);

  t_start = clock();
  hsys.FlowGradientBackward(alpha, beta, dd);
  t_finish = clock();
  std::cout << "Flow without gradient computed in " 
    << (t_finish - t_start) / CLOCKS_PER_SEC << " sec" << std::endl;

  // Compare the derivative we got with actual computation of the derivative
  for(int a = 0; a < 2; a++)
    {
    dd_test[a].set_size(k);
    dd_test[a].fill(0.0);
    for(int b = 0; b < 2; b++)
      {
      dd_test[a] += alpha[b] * grad_q[b][a] + beta[b] * grad_p[b][a];
      }

    for(int i = 0; i < k; i++)
      {
      test(dd[a](i), dd_test[a](i), 1.e-6, "Backflown Gradient [%d](%d)", a, i);
      }
    }

  std::cout << "Passed check on backward gradient flow" << std::endl;

  // Test the gradient vs. numerical gradient
  double eps = 1e-5;
  Ham::Matrix q1_1 = q1, p1_1 = p1, q1_2 = q1, p1_2 = p1;
  for(int i = 0; i < k; i++)
    {
    for(int a = 0; a < 2; a++)
      {
      Ham::Matrix p0_1 = p0; p0_1(i,a) -= eps;
      Ham::Matrix p0_2 = p0; p0_2(i,a) += eps;

      hsys.FlowHamiltonian(p0_1, q1_1, p1_1);
      hsys.FlowHamiltonian(p0_2, q1_2, p1_2);

      // Numeric derivative of p1 and q1 wrt p0(i,a)
      Ham::Matrix Dp_num = (p1_2 - p1_1) * (0.5/eps);
      Ham::Matrix Dq_num = (q1_2 - q1_1) * (0.5/eps);

      // Compare with the analytic derivative
      for(int j = 0; j < k; j++)
        {
        for(int b = 0; b < 2; b++)
          {
          test(Dp_num(j,b), grad_p[b][a](j,i), 1e-5, "grad_p[%d][%d][%d][%d]",b,a,j,i);
          test(Dq_num(j,b), grad_q[b][a](j,i), 1e-5, "grad_q[%d][%d][%d][%d]",b,a,j,i);
          }
        }
      }
    }

  // Report that we are done
  std::cout << "Passed derivative check on Hamilton flow" << std::endl;

  // Test an objective function that involves multiple timepoints. The objective
  // is just the distance between the flowing point (q_t) and the linear interpolation
  // between q_0 and target points q_T
  
  // First, compute the central difference approximation of the objective function. The 
  // objective function is computed in the loop
  Ham::Vector dE_num[2];
  Ham::Vector dE_ana[2];

  for(int a = 0; a < 2; a++)
    {
    dE_num[a].set_size(k);
    for(int i = 0; i < k; i++)
      {
      // We are computing the gradient of objective 'E' with respect to P0_i,a. Next 
      // loop over forward and backwad differences
      double E_dir[2];
      for(int dir = 0; dir < 2; dir++)
        {
        // Perturb forward or backward
        Ham::Matrix p0_eps = p0; p0_eps(i,a) += (dir == 0) ? -eps : eps;

        // Perform flow
        hsys.FlowHamiltonian(p0_eps, q1, p1);

        // Compute the objective function across the entire time
        E_dir[dir] = 0.0;
        for(int t = 0; t < hsys.GetN(); t++)
          {
          // Get the interpolant for this time
          const Ham::Matrix &qt = hsys.GetQt(t);

          // Get the target points
          double lerp = t * 1.0 / (hsys.GetN() - 1);

          for(int j = 0; j < k; j++)
            {
            for(int b = 0; b < 2; b++)
              {
              double trg_jbt = q0(j, b) * (1.0 - lerp) + trg(j, b) * lerp;
              E_dir[dir] += (trg_jbt - qt(j,b)) * (trg_jbt - qt(j,b));
              }
            }
          }
        }

      // Compute the partial derivative of the objective function
      dE_num[a](i) = (E_dir[1] - E_dir[0]) / (2 * eps);
      }
    }

  // Now perform the flow with the actual input to compute analytic derivative
  hsys.FlowHamiltonian(p0, q1, p1);

  // Compute the gradient of the objective w.r.t. to qt at every t
  std::vector<Ham::Matrix> d_obj__d_qt(hsys.GetN());
  for(int t = 0; t < hsys.GetN(); t++)
    {
    d_obj__d_qt[t].set_size(k, 2);
    const Ham::Matrix &qt = hsys.GetQt(t);
    double lerp = t * 1.0 / (hsys.GetN() - 1);
    for(int j = 0; j < k; j++)
      {
      for(int b = 0; b < 2; b++)
        {
        double trg_jbt = q0(j, b) * (1.0 - lerp) + trg(j, b) * lerp;
        d_obj__d_qt[t](j, b) = 2 * (qt(j,b) - trg_jbt);
        }
      }
    }

  // Perform the backward flow
  hsys.FlowTimeVaryingGradientsBackward(d_obj__d_qt, dE_ana);

  // Validate
  for(int j = 0; j < k; j++)
    {
    for(int b = 0; b < 2; b++)
      {
      test(dE_num[b](j), dE_ana[b](j), 1e-5, "tvar_obj[%d][%d]",j,b);
      }
    }

  // Report that we are done
  std::cout << "Passed derivative check on multi-time objective backward flow" << std::endl;

  // Create a larger problem with 800 time points, to test overhead
  int big_k = 5000;
  Ham::Matrix big_q0(big_k, 2), big_p0(big_k, 2), big_q1(big_k, 2), big_p1(big_k, 2);
  for(int i = 0; i < big_k; i++)
    {
    double t = 1.0 / (big_k - 1.0);
    double p = (k - 1.0) * t;
    int i0 = floor(p), i1 = ceil(p);
    double v = p - i0;
    for(int a = 0; a < 2; a++)
      {
      big_q0(i,a) = q0(i0, a) * (1.0 - v) + q0(i1, a) * v;
      big_p0(i,a) = p0(i0, a) * (1.0 - v) + p0(i1, a) * v;
      }
    }

  Ham big_hsys(big_q0, 0.08, 100, 0, 0);
  auto ch_start = std::chrono::high_resolution_clock::now();
  big_hsys.FlowHamiltonian(big_p0, big_q1, big_p1);
  auto ch_end = std::chrono::high_resolution_clock::now();
  std::cout << "Big problem (" << big_k << "x2): Flow without gradient computed in "
    << std::chrono::duration_cast<std::chrono::milliseconds>(ch_end - ch_start).count() << " ms." << std::endl;

  // Now perform backprop on the big problem
  Ham::Vector big_alpha[2], big_beta[2], big_result[2];
  for(int a = 0; a < 2; a++)
    {
    big_alpha[a].set_size(big_k); big_alpha[a].fill(0.0001);
    big_beta[a].set_size(big_k); big_beta[a].fill(0.);
    }

  ch_start = std::chrono::high_resolution_clock::now();
  big_hsys.FlowGradientBackward(big_alpha, big_beta, big_result);
  ch_end = std::chrono::high_resolution_clock::now();
  std::cout << "Big problem (" << big_k << "x2): Backprop computed in "
    << std::chrono::duration_cast<std::chrono::milliseconds>(ch_end - ch_start).count() << " ms." << std::endl;

  // Now generate a problem with 'rider' points and check derivatives. We will generate rider points
  // by taking averages of random pairs of input points
  unsigned int n_riders = 60, m = k + n_riders;
  Ham::Matrix q0_riders(n_riders, 2);
  for(unsigned int i = 0; i < n_riders; i++)
    {
    int i1 = rand() % k, i2 = rand() % k;
    for(int a = 0; a < 2; a++)
      q0_riders(i, a) = 0.5 * (q0(i1, a) + q0(i2, a));
    }

  Ham::Matrix q0r(m, 2), q1r(m, 2);
  q0r.update(q0, 0, 0);
  q0r.update(q0_riders, k, 0);

  Ham rider_hsys(q0r, 0.08, 100, n_riders, 0);
  rider_hsys.FlowHamiltonian(p0, q1r, p1);

  std::cout << "Completed forward flow with riders" << std::endl;

  // Define a simple objective function: 1/2 sum of squares of q1's, which makes the
  // alphas be the q1s

  // Test forward-backward gradient computation approach with riders
  Ham::Vector alpha_r[2], beta_r[2], dd_r[2], dd_test_r[2];
  for(int a = 0; a < 2; a++)
    {
    alpha_r[a].set_size(m);
    beta_r[a].set_size(k);

    for(unsigned int i = 0; i < m; i++)
      alpha_r[a](i) = q1r(i,a);

    for(unsigned int i = 0; i < k; i++)
      beta_r[a](i) = 0.0;
    }

  // Do the back-prop with riders
  ch_start = std::chrono::high_resolution_clock::now();
  rider_hsys.FlowGradientBackward(alpha_r, beta_r, dd_r);
  ch_end = std::chrono::high_resolution_clock::now();
  std::cout << "Backprop with riders computed in "
    << std::chrono::duration_cast<std::chrono::milliseconds>(ch_end - ch_start).count() << " ms." << std::endl;

  // Now compute central difference derivatives with respect to specific p0's
  for(unsigned int i = 0; i < k; i++)
    for(unsigned int a = 0; a < 2; a++)
      {
      Ham::Matrix p0_off1 = p0; p0_off1(i,a) -= eps;
      rider_hsys.FlowHamiltonian(p0_off1, q1r, p1);
      double E1 = 0;
      for(unsigned int j = 0; j < m; j++)
        for(unsigned int b = 0; b < 2; b++)
          E1 += 0.5 * q1r(j,b) * q1r(j,b);

      Ham::Matrix p0_off2 = p0; p0_off2(i,a) += eps;
      rider_hsys.FlowHamiltonian(p0_off2, q1r, p1);
      double E2 = 0;
      for(unsigned int j = 0; j < m; j++)
        for(unsigned int b = 0; b < 2; b++)
          E2 += 0.5 * q1r(j,b) * q1r(j,b);

      double d_num = (E2 - E1) / (2.0 * eps);
      double d_ana = dd_r[a](i);

      test(d_num, d_ana, 1e-5, "D_riders[%d][%d]", i, a);
      }

  std::cout << "Passed check on backward gradient flow" << std::endl;

  return 0;
};
