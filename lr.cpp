#include <iostream>
#include "lr.h"
#include <random>
#include <memory>
using namespace std;

namespace ml
{
	LinearRegression::LinearRegression(const size_t& is)
	{
		LinearRegression::input_size = is;
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<> dis(0.0, 1.0);
		weights = new VectorXd(is);
		*weights = VectorXd::Zero(is);
		for (size_t i = 0; i < is; i++)
		{
			(*weights)(i) = dis(gen);
		}
		bias = dis(gen);
	}
	VectorXd LinearRegression::getWeight()
	{
		VectorXd weightsToReturn(input_size);
		for (size_t i = 0; i < input_size; i++)
		{
			weightsToReturn(i) = (*weights)(i);
		}
		return weightsToReturn;
	}
	VectorXd LinearRegression::predict(const MatrixXd& x)
	{
		VectorXd prediction = x * (*weights) + VectorXd::Constant(x.rows(), bias);
		return prediction;
	}
	void LinearRegression::fit(const MatrixXd& x, const VectorXd& y)
	{
		*weights = (x.transpose() * x).ldlt().solve(x.transpose() * y);
		bias = (y.sum() - (x * (*weights)).sum()) / x.rows();
	}
	LinearRegression::~LinearRegression()
	{
		delete weights;
	}
}
