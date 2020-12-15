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
 * The implementation is an implementation of Algorithm 2.1 of Gaussian Processes
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
class GPRegressor
{
 public:
   
};
}
}

#endif