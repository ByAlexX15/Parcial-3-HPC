#include "regression.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

//Primera función de costo para la regresión lineal
// Basado en los minimos cuadrados ordinarios demostrado en clase

float Regression::F_OLS_Costo(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta)
{

    Eigen::MatrixXd diferencia = pow((X*theta - y).array(),2);

    return(diferencia.sum() / (2*X.rows()));

}

/*Función de gradiente descendiente: en función de un radio de
 * aprendizaje* (Learning Rate) *
 * Se avanza hasta enconctrar el punto minimo que preenta el valor óptimo para la función
 */

std::tuple<Eigen::VectorXd, std::vector<float>> Regression::GradienteDescent(Eigen::MatrixXd X,
                                                                               Eigen::MatrixXd y,
                                                                               Eigen::VectorXd theta,
                                                                               float alpha,
                                                                               int iteraciones){
    Eigen::MatrixXd temporal = theta;
    int parametros = theta.rows();
    std::vector <float> costo;
    //El costo ingresamos los valores de la función de costo
    costo.push_back(F_OLS_Costo(X,y,theta));
    //Se iteró según el número de iteraciones y el ratio de aprendizaje
    //Para encontrarlos valroes oprtimos
    for(int i = 0; i< iteraciones;i++){
        Eigen::MatrixXd error = X*theta-y;
        for(int j = 0; j< parametros;j++ ){
            Eigen::MatrixXd X_i =X.col(j);
            Eigen::MatrixXd term = error.cwiseProduct(X_i);
            temporal(j,0) = theta(j,0) - (alpha/X.rows())*term.sum();
        }
        theta = temporal;
        /*En costo ingresamos los valores de la función de costo*/
        costo.push_back(F_OLS_Costo(X,y,theta));
    }
    return std::make_tuple(theta,costo);
}

/* Acontinuación se presenta la función para revisar que tna bueno es nuestro proyecto::
 * Se procede a crear ma métrica de rendimiento:
 * R² score: coeficiente de determinación en dnde le mejor valor posible es 1*/

float Regression::r2_score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){

    auto numerador = pow((y - y_hat).array(),2).sum();
    auto denominador = pow((y.array() - y.mean()),2).sum();
    return (1 - (numerador/denominador));

}

float Regression::MSE(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){

    auto numerador1 = pow((y - y_hat).array(),2).sum();
    auto denominador1 = y.rows();
    return (numerador1/denominador1);
}

float Regression::RMSE(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){

    auto aux = sqrt(MSE(y,y_hat));
    return aux;
}
