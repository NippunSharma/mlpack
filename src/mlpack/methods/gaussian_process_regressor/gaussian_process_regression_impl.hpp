/**
 * @file methods/gaussian_process_regression/gaussian_process_regression_impl.hpp
 * @author Nippun Sharma
 *
 * Implementation of the GPRegressor class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
**/

#ifndef MLPACK_METHODS_GAUSSIAN_PROCESS_REGRESSION_GAUSSIAN_PROCESS_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_GAUSSIAN_PROCESS_REGRESSION_GAUSSIAN_PROCESS_REGRESSION_IMPL_HPP

#include "gaussian_process_regression.hpp"

namespace mlpack {
namespace regression {

template<typename KernelType>
GPRegressor<KernelType>::GPRegressor(
    const KernelType kernel,
    const size_t alpha = 1e-7,
    const bool normalize_y = false) :
    kernel(kernel),
    alpha(alpha),
    normalize_y(normalize_y)
{
  // Nothing to do here.
}

template<typename KernelType>
double GPRegressor<KernelType>::Train(
    const arma::mat& data,
    const arma::rowvec& respones)
{

}

template<typename KernelType>
void GPRegressor<KernelType>::Predict(
    const arma::mat& points,
    const arma::rowvec& predictions) const
{

}

template<typename KernelType>
void GPRegressor<KernelType>::Predict(
    const arma::mat& points,
    const arma::rowvec& predictions,
    const arma::rowvec& std) const
{

}

template<typename KernelType>
double GPRegressor<KernelType>::RMSE(
    const arma::mat& data,
    const arma::rowvec& responses) const
{

}

}
}

#endif