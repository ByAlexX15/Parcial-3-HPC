/* Fecha: 21 sept 2022
 * Autor; Rubén Alexis Núñez
 * Materia: HPC
 * Tópico : IMplementación de las Regresión Lineal como modelo en C++
 * Requerimientos:
 *  - COnstriur una clase Extracción, que permita
 *  manipular, extraer y cargar los datos.
 *  - Construir una clase LinearRegression, que permita
 *  los calculos de la función de costo, gradientes descendiente
 *  entre otras
 *
 * */

#include "ClassExtraction/extractiondata.h"
#include "Regression/regression.h"
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <list>
#include <vector>
#include <fstream>
#include <iostream>

int main(int argc, char* argv[])
{

    //Se crea un objeto del tipo ClassExtraction
    ExtractionData ExData(argv[1],argv[2],argv[3]);
    //SE instancia la clase de regresión lineal en un objeto

    Regression modeloLR;

    //Se crea un vector de vectores del tipo string para cargar objeto ExData
    std::vector<std::vector<std::string>> dataframe = ExData.LeerCSV();

    //Cantidad de filas y columnas
    int filas    = dataframe.size();
    int columnas = dataframe[0].size();

    //Se crea una matriz Eigen, para ingresar los valores a esa matriz
    Eigen::MatrixXd matData = ExData.CSVtoEigen(dataframe, filas, columnas);


    /*Se normaliza la matriz de los datos */
    Eigen::MatrixXd matNorm = ExData.Norm(matData);


    /*Se divide en datos de entrenamiento y datos de prueba*/
    Eigen::MatrixXd X_train, y_train, X_test, y_test;


    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> tupla_datos = ExData.TrainTestSplit(matNorm, 0.8);
    /*Se descomprime la tupla en 4 conjuntos */

    std::tie(X_train,y_train,X_test,y_test) = tupla_datos;

    /* Se crea vectores auxiliares para prueba y entrenamiento inicializados en 1 */
    Eigen::VectorXd vector_train = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vector_test = Eigen::VectorXd::Ones(X_test.rows());

    /* Se redimensiona la matriz de entrenamiento y de prueba para ser ajustadas a
     * los vectores auxiliares anteriores */
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vector_train;
    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vector_test;

    /* Se crea el vector de coeficientes theta */
    Eigen::VectorXd thetas =Eigen::VectorXd::Zero(X_train.cols());
    /* Se establece el alpha como ratio de aprendizaje de tipo flotante */
    float alpha = 0.01;
    int num_iter = 1000; //num iteraciones

    //Se crea un vector para almacenar las thetas de salida (parametro)
    Eigen::MatrixXd thetas_out;
    //Se crea un vector sencillo(std) de flotantes para almacenar los valores del costo
    std::vector<float> costo;
    //Se calcula el gradeinte descendiente
    std::tuple<Eigen::VectorXd, std::vector<float>> g_descendiente = modeloLR.GradienteDescent(X_train,
                                                                                               y_train,
                                                                                               thetas,
                                                                                               alpha,
                                                                                               num_iter);

    //Se desempaqueta el gradiente
    std::tie(thetas_out,costo) = g_descendiente;

     /*SE almacena los valores de thetas y costos en un fichero para psoteriormente ser visualizados */

    ExData.VectortoFile(costo, "costo.txt");
    ExData.EigentoFile(thetas_out, "thetas.txt");

    /*SE extrae el promedio de la matriz de entrada */
    auto prom_data = ExData.Promedio(matData);
    //Se extraen los valores de la varibales independientes
    auto var_prom_independientes = prom_data(0,8);
    //SE escalan los datos
    auto datos_escalados = matData.rowwise() - matData.colwise().mean();
    //Se extrae la desviación estandar de datos escalados
    auto dev_stand = ExData.DevStand(datos_escalados);
    //Se extrane los valores de las viarbales independientes de la desviación estandar
    auto var_des_independientes =dev_stand(0,8);
    //Se crea una maitriz para almacenar los valores estimados de entrenamiento.
    Eigen::MatrixXd y_train_hat = (X_train * thetas_out * var_des_independientes).array() + var_prom_independientes;
    //Matriz para los valores reales de y
    Eigen::MatrixXd y = matData.col(8).topRows(1599);
    //Para la prueba
    Eigen::MatrixXd y_test_hat = (X_test * thetas_out * var_des_independientes).array() + var_prom_independientes;
    Eigen::MatrixXd y_test1 = matData.col(8).bottomRows(400);
    //Se revisa que tan bueno quedó el modelo a traves de una metrica de rendimiento
    float metrica_R2 = modeloLR.r2_score(y_test1,y_test_hat);
    float metrica_R22 = modeloLR.r2_score(y,y_train_hat);
    float metrica_MSE = modeloLR.MSE(y, y_train_hat);
    float metrica_RMSE = modeloLR.RMSE(y, y_train_hat);

    std::cout<<"ENTRENAMIENTO"<<std::endl;

    std::cout << "metrica_R2 en train " <<metrica_R22 <<std::endl;
    std::cout<<"Matriz de entrenamiento, número " << X_train.rows() << " de Filas \n"<<std::endl;

    std::cout<<"Metrica MSE "<< metrica_MSE<<std::endl;
    std::cout<<"Metrica RMSE "<< metrica_RMSE<<std::endl;

    std::cout<<"TEST"<<std::endl;
    std::cout << "metrica_R2 en test " <<metrica_R2 <<std::endl;
    std::cout<<"Matriz de prueba, número " << X_test.rows() << " de Filas \n"<<std::endl;

    std::cout<<"Información general"<<std::endl;
    std::cout<<"Matriz total, número " << matData.rows() << " de Filas \n"<<std::endl;
    std::cout<<"El Promedio de la matriz " << ExData.Promedio(matData) << "\n"<<std::endl;
    std::cout<<"La desvación estandar de la matriz es de:"<<ExData.DevStand(datos_escalados)<<""<<std::endl;



    return EXIT_SUCCESS;
}
