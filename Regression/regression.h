#ifndef REGRESSION_H
#define REGRESSION_H
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

class Regression
{
public:
    float F_OLS_Costo(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd, std::vector<float>> GradienteDescent(Eigen::MatrixXd X,
                                                                     Eigen::MatrixXd y,
                                                                     Eigen::VectorXd theta,
                                                                     float alpha,
                                                                     int iteraciones);
    float r2_score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);

    float MSE(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
    float RMSE(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);

};

#endif // REGRESSION_H
