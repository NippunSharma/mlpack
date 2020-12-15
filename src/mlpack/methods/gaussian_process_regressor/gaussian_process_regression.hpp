/**
 * @file methods/gaussian_process_regression/gaussian_process_regression.hpp
 * @author Nippun Sharma
 *
 * Definition of the GPRegressor class that uses gaussian process regression
 * for prediction. According to armadillo standards, all functions consider data in 
 * column-major format.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
**/

#ifndef MLPACK_METHODS_GAUSSIAN_PROCESS_REGRESSION_GAUSSIAN_PROCESS_REGRESSION_HPP
#define MLPACK_METHODS_GAUSSIAN_PROCESS_REGRESSION_GAUSSIAN_PROCESS_REGRESSION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack{
namespace regression{

/**
 * This is an implementation of Algorithm 2.1 of Gaussian Processes
 * for Machine Learning (GPML) by Rasmussen and Williams.
 * 
 * Example of use:
 * 
 * @code
 * arma::mat x_train; // Train data matrix, Column-major format.
 * arma::mat y_train; // Train target values.
 * 
 * // Train the model. Kernel hyperparameters are tuned automatically.
 * // while training.
 * GPRegressor model();
 * model.Train(x_train, y_train);
 * 
 * // Prediction on test points.
 * arma::mat x_test;
 * arma::rowvec predictions;
 * 
 * model.Predict(x_test, predictions);
 * 
 * arma::row_vec y_test; // test target values.
 * model.RMSE(x_test, y_test); // Evaluate using RMSE score.
 * 
 * // Compute the standard deviations of the predictions.
 * arma::rowvec stds;
 * model.Predict(x_test, predictions, stds);
 * @endcode
 */
template<typename KernelType>
class GPRegressor
{
 public:
  /**
   * Set the parameters for the GPRegressor object. The kernel hyperparameters
   * are automatically optimised by maximizing the log marginal likelihood.
   * 
   * @param kernel Kernel used for covariance matrix.
   * @param alpha Value added to the diagnol of kernel matrix.
   * @param normalize_y Whether the target values are normalized while training.
   */
  GPRegressor(const KernelType kernel = KernelType(),
              const size_t alpha = 1e-7,
              const bool normalize_y = false);

  /**
   * Run the GPRegressor. The input matrix (like all mlpack matrices) must be
   * in column-major format i.e. each column must be an observation and each
   * row must be a dimension.
   * 
   * @param data Column-major input data, dim(P,N).
   * @param responsed A vector of targets, dim(N).
   */
  double Train(const arma::mat& data,
               const arma::rowvec& responses);

  /**
   * Predict for each data point in the given data matrix using
   * currently trained GPRegressor model.
   * 
   * @param points, The data points to apply the model.
   * @param predictions, Will contain the predicted values on completion.
   */
  void Predict(const arma::mat& points,
               const arma::rowvec& predictions) const;

  /**
   * Predict for each data point and also get the standard deviation of
   * the predictive posterior distribution for each data point in the given
   * data matrix, using the currently trained GPRegressor.
   * 
   * @param points, The data points to apply the model.
   * @param predictions, Will contain the predicted values on completion.
   * @param std, Will contain the standard deviations on completion.
   */
  void Predict(const arma::mat& points,
               const arma::rowvec& predictions,
               const arma::rowvec& std) const;

  /**
   * Compute the Root Mean Squared Error b/w predictions returned by the model
   * an the true values.
   * 
   * @param data, Data points to predict.
   * @param responses, A vector of targets.
   * @return Root Mean Squared Error.
   */
  double RMSE(const arma::mat& data,
              const arma::rowvec& responses) const;
 private:
  //! KernelType used for covariance matrix.
  KernelType kernel;

  //! Small value to add to diagnol of covariance matrix to
  //! represent noise or to add numerical stability.
  size_t alpha;

  //! Whether to normalize the target values or not.
  bool normalize_y;

  //! Center data if true.
  bool centerData;

  //! Scale data with standard deviations if true.
  bool scaleData;

  //! Maximum number of iterations for convergence.
  size_t maxIter;

  //! Tolerance for which solution is considered stable.
  double tol;

  //! Mean vector computed over points.
  arma::colvec dataOffset;

  //! Mean of the response vector calculated over points.
  double responsesOffset;

  //! Solution vector.
  arma::colvec omega;

  //! Covariance matrix of solution vector omega.
  arma::mat matCovariance;

};
}
}

// Include implementation
#include "gaussian_process_regression_impl.hpp"

#endif