/**
 * @file core/data/load_csv.hpp
 * @author ThamNgapWei
 *
 * This is a csv parsers which use to parse the csv file format
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_CSV_HPP
#define MLPACK_CORE_DATA_LOAD_CSV_HPP

#include "rapidcsv.h"

#include <mlpack/core.hpp>
#include <mlpack/core/util/log.hpp>

#include <set>
#include <string>

#include "extension.hpp"
#include "format.hpp"
#include "dataset_mapper.hpp"

namespace mlpack {
namespace data {

/**
 *Load the csv file.This class use boost::spirit
 *to implement the parser, please refer to following link
 *http://theboostcpplibraries.com/boost.spirit for quick review.
 */
class LoadCSV
{
 public:
  /**
   * Construct the LoadCSV object on the given file.  This will construct the
   * rules necessary for loading and attempt to open the file.
   */
  LoadCSV(const std::string& file);

  /**
   * Load the file into the given matrix with the given DatasetMapper object.
   * Throws exceptions on errors.
   *
   * @param inout Matrix to load into.
   * @param infoSet DatasetMapper to use while loading.
   * @param transpose If true, the matrix should be transposed on loading
   *     (default).
   */
  template<typename T, typename PolicyType>
  void Load(arma::Mat<T> &inout,
            DatasetMapper<PolicyType> &infoSet,
            const bool transpose = true)
  {
    CheckOpen();

    doc = rapidcsv::Document(filename, rapidcsv::LabelParams(-1, -1));

    if (transpose)
      TransposeParse(inout, infoSet);
    else
      NonTransposeParse(inout, infoSet);
  }

  /**
   * Peek at the file to determine the number of rows and columns in the matrix,
   * assuming a non-transposed matrix.  This will also take a first pass over
   * the data for DatasetMapper, if MapPolicy::NeedsFirstPass is true.  The info
   * object will be re-initialized with the correct dimensionality.
   *
   * @param rows Variable to be filled with the number of rows.
   * @param cols Variable to be filled with the number of columns.
   * @param info DatasetMapper object to use for first pass.
   */
  template<typename T, typename MapPolicy>
  void GetMatrixSize(size_t& rows, size_t& cols, DatasetMapper<MapPolicy>& info)
  {
    // Take a pass through the file.  If the DatasetMapper policy requires it,
    // we will pass everything string through MapString().  This might be useful
    // if, e.g., the MapPolicy needs to find which dimensions are numeric or
    // categorical.

    // Reset to the start of the file.
    rows = doc.GetRowCount();
    cols = doc.GetColumnCount();

    // First, count the number of rows in the file (this is the dimensionality).
    info = DatasetMapper<MapPolicy>(cols);

    // Now, jump back to the beginning of the file.

    size_t counter = 0;
    if (MapPolicy::NeedsFirstPass)
    {
      while (cols - counter > 0)
      {
        std::vector<std::string> temp = doc.GetColumn<std::string>(counter);
        for (int l=0; l<temp.size(); ++l)
        {
          info.template MapFirstPass<T>(temp[l], counter);
        }
        ++counter;
      }
    }
  }

  /**
   * Peek at the file to determine the number of rows and columns in the matrix,
   * assuming a transposed matrix.  This will also take a first pass over the
   * data for DatasetMapper, if MapPolicy::NeedsFirstPass is true.  The info
   * object will be re-initialized with the correct dimensionality.
   *
   * @param rows Variable to be filled with the number of rows.
   * @param cols Variable to be filled with the number of columns.
   * @param info DatasetMapper object to use for first pass.
   */
  template<typename T, typename MapPolicy>
  void GetTransposeMatrixSize(size_t& rows,
                              size_t& cols,
                              DatasetMapper<MapPolicy>& info)
  {
    // Take a pass through the file.  If the DatasetMapper policy requires it,
    // we will pass everything string through MapString().  This might be useful
    // if, e.g., the MapPolicy needs to find which dimensions are numeric or
    // categorical.

    // Reset to the start of the file.
    rows = doc.GetColumnCount();
    cols = doc.GetRowCount();

    // First, count the number of rows in the file (this is the dimensionality).
    info = DatasetMapper<MapPolicy>(cols);

    // Now, jump back to the beginning of the file.

    size_t counter = 0;
    if (MapPolicy::NeedsFirstPass)
    {
      while (cols - counter > 0)
      {
        std::vector<std::string> temp = doc.GetRow<std::string>(counter);
        for (int l=0; l<temp.size(); ++l)
        {
          info.template MapFirstPass<T>(temp[l], counter);
        }
        ++counter;
      }
    }
  }

 private:

  /**
   * Check whether or not the file has successfully opened; throw an exception
   * if not.
   */
  void CheckOpen();

  /**
   * Parse a non-transposed matrix.
   *
   * @param inout Matrix to load into.
   * @param infoSet DatasetMapper object to load with.
   */
  template<typename T, typename PolicyType>
  void NonTransposeParse(arma::Mat<T>& inout,
                         DatasetMapper<PolicyType>& infoSet)
  {
    // Get the size of the matrix.
    size_t rows, cols;
    GetMatrixSize<T>(rows, cols, infoSet);

    // Set up output matrix.
    inout.set_size(rows, cols);
    size_t row = 0;
    size_t col = 0;
    while (row < rows)
    {
      std::vector<std::string> temp = doc.GetRow<std::string>(row);
      for (int l=0; l<temp.size(); ++l)
      {
        inout(row, col) = infoSet.template MapString<T>(temp[l], col);
        col += 1;
      }

      // Make sure we got the right number of rows.
      if (col != cols)
      {
        std::ostringstream oss;
        oss << "LoadCSV::NonTransposeParse(): wrong number of dimensions ("
            << col << ") on line " << row << "; should be " << cols
            << " dimensions.";
        throw std::runtime_error(oss.str());
      }

      /*
      if (!canParse)
      {
        std::ostringstream oss;
        oss << "LoadCSV::NonTransposeParse(): parsing error on line " << col
            << "!";
        throw std::runtime_error(oss.str());
      }
      */
      ++row; col = 0;
    }
  }

  /**
   * Parse a transposed matrix.
   *
   * @param inout Matrix to load into.
   * @param infoSet DatasetMapper to load with.
   */
  template<typename T, typename PolicyType>
  void TransposeParse(arma::Mat<T>& inout, DatasetMapper<PolicyType>& infoSet)
  {
    // Get matrix size.  This also initializes infoSet correctly.
    size_t rows, cols;
    GetTransposeMatrixSize<T>(rows, cols, infoSet);

    // Set the matrix size.
    inout.set_size(rows, cols);

    // Initialize auxiliary variables.
    size_t row = 0;
    size_t col = 0;

    while (col < cols)
    {
      std::vector<std::string> temp = doc.GetRow<std::string>(col);
      for (int l=0; l<temp.size(); ++l)
      {
        inout(row, col) = infoSet.template MapString<T>(temp[l], col);
        row += 1;
      }
      // Make sure we got the right number of rows.
      if (row != rows)
      {
        std::ostringstream oss;
        oss << "LoadCSV::TransposeParse(): wrong number of dimensions (" << row
            << ") on line " << col << "; should be " << rows << " dimensions.";
        throw std::runtime_error(oss.str());
      }
      /*
      if (!canParse)
      {
        std::ostringstream oss;
        oss << "LoadCSV::TransposeParse(): parsing error on line " << col
            << "!";
        throw std::runtime_error(oss.str());
      }
      */
      // Increment the column index and reset row index.
      ++col;row=0;
    }
  }

  rapidcsv::Document doc;

  //! Extension (type) of file.
  std::string extension;
  //! Name of file.
  std::string filename;
  //! Opened stream for reading.
  std::ifstream inFile;
};

} // namespace data
} // namespace mlpack

#endif
