#include <iostream>
#include "pr.h"
#include <Eigen/Dense>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace ml
{
	PolynomialRegression::PolynomialRegression(const size_t degree, const size_t input)
	{
		PolynomialRegression::degree = degree;
		PolynomialRegression::input_len = input;
		weights = new MatrixXd(input, degree);
	}
	void PolynomialRegression::fit(const MatrixXd& x, const VectorXd& y)
	{
		MatrixXd newX = this->translate(x);
		size_t n = x.rows();
		VectorXd newY(n);
		for (size_t i = 0; i < n; ++i)
		{
			newY(i) = y(i);
		}
		*weights = (newX.transpose() * newX).ldlt().solve(newX.transpose() * newY);
	}
	VectorXd PolynomialRegression::predict(const MatrixXd& x)
	{
		VectorXd y = this->translate(x) * (*weights);
		return y;
	}
	MatrixXd PolynomialRegression::getWeights()
	{
		MatrixXd w = *weights;
		return w;
	}
	MatrixXd PolynomialRegression::translate(const MatrixXd& x)
	{
		size_t n = x.rows();
		size_t d = x.cols();
		size_t num_f = 1;
		for (size_t i = 1; i <= degree; ++i)
		{
			num_f += pow(d, i);
		}
		MatrixXd newX(n, num_f);
		for (size_t i = 0; i < n; ++i)
		{
			size_t col = 0;
			newX(i, col) = 1;
			for (size_t j = 1; j <= degree; ++j)
			{
				for (size_t k = 0; k < d; ++k)
				{
					newX(i, col++) = pow(x(i, k), j);
				}
			}
		}
		return newX;
	}
	PolynomialRegression::~PolynomialRegression()
	{
		delete weights;
	}
}