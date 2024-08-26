#pragma once
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace ml
{
	class PolynomialRegression
	{
	private:
		MatrixXd* weights;
		size_t degree;
		size_t input_len;
		MatrixXd translate(const MatrixXd& x);

	public:
		PolynomialRegression(size_t degree, size_t input);
		VectorXd predict(const MatrixXd& x);
		void fit(const MatrixXd& x, const VectorXd& y);
		MatrixXd getWeights();
		~PolynomialRegression();
	};
}