#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <memory>
using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace ml
{
	class LinearRegression
	{
	private:
		size_t input_size;
		VectorXd* weights;
		double bias;
	public:
		LinearRegression(const size_t& is);
		VectorXd getWeight();
		VectorXd predict(const MatrixXd& x);
		void fit(const MatrixXd& x, const VectorXd& y);
		~LinearRegression();
	};
}