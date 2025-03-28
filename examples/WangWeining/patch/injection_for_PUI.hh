#pragma once
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <random>

const double alpha_PUI = 1.4;
const double r_PUI = 33.5;
const double Vc = 10.07;
const double eta = 5;
const double lambda = 3.4;
// 阶跃函数
double stepFunction(double x) {
    return (x < 0)? 0.0 : 1.0;
}

// 定义任意的速度分布函数 f(v)
double f(double v) {
    const double x = v / Vc;
    return (x <= 0)? 0.0 : std::pow(x, alpha_PUI - 3) * std::exp(-lambda / r_PUI * std::pow(x, -alpha_PUI)) * stepFunction(1 - x);
}
double speed_pdf(double x) {  
    // 
    double lambda = 3.4;
    
    if (x <= 0) return 0.0;  
    return 4*M_PI*pow(x,alpha_PUI-1)*exp(-lambda/r_PUI*pow(x,-alpha_PUI))*stepFunction(1-x) ; // 
}  
double speed_cdf(double x) {  
    // 
    //double lambda = 3.4;
    double delta_x = 0.0005;
    double S = 0;
  
    for (int i = 0; i < floor(x/delta_x); ++i){
        S = S + speed_pdf(i*delta_x)*delta_x;
    }
    if (x <= 0) return 0.0;  
    return S; // 
} 
double inverse_cdf(double y, double tol = 1e-3) {  
    double low = 0;             
    double high = 1;  
    double mid; 
    // double alpha = 1.4;
    // double eta = 5, r = 33.5;
    double n = speed_cdf(1);
    // std::cout<<n<<"\n";
    while (high - low > tol) {  
        mid = (low + high) / 2.0;  

        if (speed_cdf(mid)/n < y) {  
            low = mid;  
        } else {  
            high = mid;  
        }  
    }  
    return (low + high) / 2.0;  
} 
// 计算二维柱坐标速度空间的积分（考虑到 y,z 方向各项同性）
double integral_flux(double vd, double dvx, double dvr) {
    double sum = 0;
    const double vx_start = std::max(0.0, vd-Vc);
    const double vx_end = vd + Vc;
    for (double vx = vx_start; vx <= vx_end; vx += dvx) {
        if ((vx-vd)*(vx-vd)>Vc*Vc) continue;
        double vr_max = sqrt(Vc*Vc-(vx-vd)*(vx-vd));
        for (double vr = 0; vr <= vr_max; vr += dvr) {
            double v = std::sqrt((vx - vd) * (vx - vd) + vr * vr);
            // 在柱坐标系中，面积微元为 2πvr * dvr * dvx
            sum += f(v) * vx * 2 * M_PI * vr * dvr * dvx; 
        }
    }
    return sum;
}
// double integral_f(double vd, double dvx, double dvr) {
//     double sum = 0;
//     const double vx_start = std::max(0.0, vd-Vc);
//     const double vx_end = vd + Vc;
//     for (double vx = vx_start; vx <= vx_end; vx += dvx) {
//         if ((vx-vd)*(vx-vd)>Vc*Vc) continue;
//         double vr_max = sqrt(Vc*Vc-(vx-vd)*(vx-vd));
//         for (double vr = 0; vr <= vr_max; vr += dvr) {
//             double v = std::sqrt((vx - vd) * (vx - vd) + vr * vr);
//             // 在柱坐标系中，面积微元为 2πvr * dvr * dvx
//             sum += f(v) * vx * 2 * M_PI * vr * dvr * dvx; 
//         }
//     }
//     return sum;
// }
double g(double vx, double vr, double vd, double denominator){
    double v = sqrt((vx-vd)*(vx-vd)+vr*vr);
    return vx*f(v)/denominator;

}
// 计算累积概率分布函数 F(v)
double F(double v, double vd, double denominator, double dvx, double dvr) {
    double sum = 0;
    double vx_max = vd+v;
    // double denominator = integral_f(vd, vx_max, vr_max, dvx, dvr);
    for (double vx = 0; vx <= v; vx += dvx) {
        //std::cout<<vx<<"\n";
        if ((vx-vd)*(vx-vd)>Vc*Vc) continue;
        double vr_max = sqrt(Vc*Vc-(vx-vd)*(vx-vd));
        for (double vr = 0; vr <= vr_max; vr += dvr) {
            double v_current = std::sqrt((vx - vd) * (vx - vd) + vr * vr);
            
            sum += f(v_current) * vx * 2 * M_PI * vr * dvr * dvx;
        }
    }
    return sum / denominator;
}
// 提议分布 q(v)，在柱坐标系下
double q_cylindrical(double vd) {
    const double vx_start = std::max(0.0, vd-Vc);
    const double vx_end = vd + Vc;
    const double vr_start = 0;
    const double vr_end = vd + Vc;
    double volume = (vx_end - vx_start) * (vr_end * vr_end - vr_start * vr_start) * M_PI;
    return 1.0 / volume;
}

// 拒绝采样生成随机速度，在柱坐标系下
std::vector<double> rejection_sampling_cylindrical(double vd, double M, double denominator) {
    const double vx_start = std::max(0.0, vd-Vc);
    const double vx_end = vd + Vc;
    const double vr_start = 0;
    const double vr_end = vd + Vc;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(vx_start, vx_end);
    std::uniform_real_distribution<> dis_r(vr_start, vr_end);
    std::uniform_real_distribution<> dis_u(0, 1);

    while (true) {
        double vx = dis_x(gen);
        double vr = dis_r(gen);
        double u = dis_u(gen) * M * q_cylindrical(vd);

        if (u <= g(vx, vr, vd, denominator)) {
            std::random_device rd_theta;
            std::mt19937 gen_theta(rd_theta());
            std::uniform_real_distribution<> dis_theta(0, 2 * M_PI);
            double theta = dis_theta(gen_theta);
            double vy = vr * std::cos(theta);
            double vz = vr * std::sin(theta);
            return {vx, vy, vz};
        }
    }
}
// 使用二分法找到 F 的逆函数 F^{-1}(u)
double inverse_F(double u, double vd, double dvx, double dvr) {
    double low = 0;
    double high = Vc;
    double precision = 1e-2;
    double mid;
    const double denominator = integral_flux(vd, dvx, dvr);
    while (high - low > precision) {
        //std::cout<<high-low<<"\n";
        mid = (low + high) / 2;
        double F_value = F(mid, vd, denominator, dvx, dvr);
        if (F_value < u) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return low;
}

// 生成服从 (0, 1) 均匀分布的随机数
double uniform_random() {
    return static_cast<double>(std::rand()) / RAND_MAX;
}

