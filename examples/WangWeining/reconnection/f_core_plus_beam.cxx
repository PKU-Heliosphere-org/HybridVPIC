#include <iostream>
#include <random>
#include <vector>

// 假设这里有两个种类（c和b）的双麦氏分布参数
// 以平行方向参数为例，其他方向参数类似
const double n_c = 1.0; 
const double n_b = 0.5; 
// const double T_parallel_c = 1;
const double U_parallel_c = 0; 
const double U_parallel_b = -1.72; 
const double U_perp_c = 0; 
const double U_perp_b = 0.0; 
const double vth_parallel_c = 1.0; 
const double vth_parallel_b = 1.25; 
const double vth_perp_c = 1.0; 
//const double vth_perp_b = 1.25;
// 生成满足双麦氏分布在平行方向的随机数
double generateRandom(double U_c, double U_b, double vth_c, double vth_b) {
    std::random_device rd;  // 硬件随机数生成器
    std::mt19937 gen(rd());  // 标准的Mersenne Twister引擎

    std::normal_distribution<> dist_c(U_c, vth_c);
    std::normal_distribution<> dist_b(U_b, vth_b);

    // 以n_c和n_b的比例随机选择一种分布来生成随机数
    std::uniform_real_distribution<> dist_selector(0.0, n_c + n_b);
    double selector = dist_selector(gen);
    if (selector < n_c) {
        return dist_c(gen);
    } else {
        return dist_b(gen);
    }
}

