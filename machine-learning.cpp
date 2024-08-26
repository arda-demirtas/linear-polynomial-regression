#include <iostream>
#include "lr.h"
#include <Eigen/Dense>
#include "pr.h"

using namespace std;
int main()
{
	/*
	ml::LinearRegression lr(1);
	MatrixXd x(10, 1);
	VectorXd y(10);

	x << 1,
		2,
		3,
		4,
		5,
		6,
		7,
		8,
		9,
		10;

	y << 2,
		4,
		6,
		8,
		10,
		12,
		14,
		16,
		18,
		20;
	lr.fit(x, y);
	MatrixXd test(2, 1);
	test(0, 0) = 12;
	test(1, 0) = 24;
	cout << lr.predict(test) << endl;
	*/
	MatrixXd x(10, 1);
	x << 1, 
		2, 
		3, 
		4, 
		5, 
		6, 
		7, 
		8, 
		9, 
		10;
	VectorXd y(10);
	y << 1, 
		4, 
		9, 
		16, 
		25,
		36,
		49, 
		64, 
		81, 
		100;
	ml::PolynomialRegression pr(2, 1);
	pr.fit(x, y);
	MatrixXd test(2, 1);
	test << 2, 
			5;
	cout << pr.predict(test) << endl;
}

