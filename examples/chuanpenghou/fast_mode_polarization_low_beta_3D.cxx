
struct PerturbationParams {
    std::vector<double> amplitude_ratio;
    std::vector<double> kx_random;
    std::vector<double> ky_random;
    std::vector<double> kz_random;
    std::vector<double> phi_random;
    int n_modes;
    double n0;
    double Cs;
    double va;
    double b0x;
    double b0y;
    double b0z;
    double waveamp;
};

// 计算向量的模
double vector_norm(const std::vector<double>& v) {
    if (v.size() != 3) {
        throw std::invalid_argument("vector_norm requires a vector of size 3");
    }
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
// 计算两个向量的叉积
std::vector<double> cross_product(const std::vector<double>& a, const std::vector<double>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}
// 计算两个向量夹角的正弦值
double sine_of_angle(const std::vector<double>&  a, const std::vector<double>&  b) {
    std::vector<double> cross = cross_product(a, b);
    double cross_norm = vector_norm(cross);
    double a_norm = vector_norm(a);
    double b_norm = vector_norm(b);
    return cross_norm / (a_norm * b_norm);
}


// double get_param_M(const PerturbationParams& params) {
//     double v1x = params.v1y * get_ratio(cs2va2, cos_kB, sin_kB, params)
//     return params.b0x*params.v1y - params.v1x*params.b0y;
// }

double get_param_Vpf(double cs2va2, double cos_kB, const PerturbationParams& params) {
    return sqrt((cs2va2 + sqrt(cs2va2 * cs2va2 - 4 * params.Cs * params.Cs * params.va * params.va * cos_kB * cos_kB)) / 2);
}

double get_ratio(int i, const  PerturbationParams& params){
    std::vector<double> k = {params.kx_random[i], params.ky_random[i], params.kz_random[i]};
    std::vector<double> B0 = {params.b0x, params.b0y, params.b0z};
    double norm_b0 = sqrt(params.b0x * params.b0x + params.b0y * params.b0y + params.b0z * params.b0z);
    double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i] + params.kz_random[i] * params.kz_random[i]);
    double cos_kB = (params.kz_random[i] * params.b0z) / norm_k / norm_b0;
    double sin_kB = sine_of_angle(k, B0);
    double cs2va2 = params.Cs * params.Cs + params.va * params.va;
    double Vpf = get_param_Vpf(cs2va2, cos_kB, params);
    double alpha = params.va * params.va + params.Cs * params.Cs * sin_kB * sin_kB;
    double delta = params.Cs * params.Cs * cos_kB * sin_kB;
    if (abs(cos_kB) > 0.9996) {
        return 0.0;
    } else if (abs(cos_kB) < 0.0002) {
	    return 0.0;
    } else {
        return (Vpf * Vpf - alpha) / delta;
    } 
}

double BX_PERT(double x, double y, double z, const PerturbationParams& params) {
    double bx_pert = 0.0;
    double cs2va2 = params.Cs * params.Cs + params.va * params.va;
    double norm_b0 = sqrt(params.b0x * params.b0x + params.b0y * params.b0y + params.b0z * params.b0z);

    for (int i = 0; i < params.n_modes; i++) {
        double norm_k_perp = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i]);
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i] + params.kz_random[i] * params.kz_random[i]);
        double cos_kB = (params.kz_random[i] * params.b0z) / norm_k / norm_b0;
        double Vpf = get_param_Vpf(cs2va2, cos_kB, params);
        double v1x;
        if (norm_k_perp < 1e-6) {
            v1x = params.waveamp;
        } else {
            v1x = params.kx_random[i] / norm_k_perp * params.waveamp;
        }
        double phi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        double omega = norm_k * Vpf;
        bx_pert += params.amplitude_ratio[i] * v1x * params.b0z * params.kz_random[i] / omega * cos(phi);
    }
    return bx_pert;
}

double BY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double by_pert = 0.0;
    double cs2va2 = params.Cs * params.Cs + params.va * params.va;
    double norm_b0 = sqrt(params.b0x * params.b0x + params.b0y * params.b0y + params.b0z * params.b0z);
    
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k_perp = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i])+1e-10;
        std::vector<double> k = {params.kx_random[i], params.ky_random[i], params.kz_random[i]};
        std::vector<double> B0 = {params.b0x, params.b0y, params.b0z};
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i] + params.kz_random[i] * params.kz_random[i]);
        double cos_kB = (params.kz_random[i] * params.b0z) / norm_k / norm_b0;
        double sin_kB = sine_of_angle(k, B0);
        double Vpf = get_param_Vpf(cs2va2, cos_kB, params);
        double v1y;
        if (norm_k_perp < 1e-6) {
            v1y = 0.0;
        } else {
            v1y = params.ky_random[i] / norm_k_perp * params.waveamp;
        }
        double phi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        by_pert +=  - params.amplitude_ratio[i] * v1y * params.b0z * params.kz_random[i] / Vpf / norm_k * cos(phi);
    }
    return by_pert;
}

double BZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double bz_pert = 0.0;
    double cs2va2 = params.Cs * params.Cs + params.va * params.va;
    double norm_b0 = sqrt(params.b0x * params.b0x + params.b0y * params.b0y + params.b0z * params.b0z);

    // double param_M = get_param_M(params);
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i] + params.kz_random[i] * params.kz_random[i]);
        double norm_k_perp = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i])+1e-10;
        double cos_kB = (params.kz_random[i] * params.b0z) / norm_k / norm_b0;
        double v1x, v1y;
        double Vpf = get_param_Vpf(cs2va2, cos_kB, params);
        if (norm_k_perp < 1e-6) {
            v1x = params.waveamp;
            v1y = 0.0;
        } else {
            v1x = params.kx_random[i] / norm_k_perp * params.waveamp;
            v1y = params.ky_random[i] / norm_k_perp * params.waveamp;
        }
        double omega = norm_k * Vpf;
        double phi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        bz_pert += params.amplitude_ratio[i] * (v1x * params.b0z * params.kx_random[i] + v1y * params.b0z * params.ky_random[i]) / omega * cos(phi);
    }
    return bz_pert;
}

double UX_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ux_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k_perp = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i])+1e-10;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        double v1x;
        if (norm_k_perp < 1e-6) {
            v1x = params.waveamp;
        } else {
            v1x = params.kx_random[i] / norm_k_perp * params.waveamp;
        }
        double phi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        ux_pert += params.amplitude_ratio[i] * v1x * cos(psi); // 示例计算，仅基于 z 和 kx_random
    }
    return ux_pert;
}

double UY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double uy_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k_perp = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i])+1e-10;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        double v1y;
        if (norm_k_perp < 1e-6) {
            v1y = 0.0;
        } else {
            v1y = params.ky_random[i] / norm_k_perp * params.waveamp;
        }
        uy_pert += params.amplitude_ratio[i] * v1y * cos(psi); // 示例计算，仅基于 z 和 kx_random
    }
    return uy_pert;
}

double UZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double uz_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k_perp = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i])+1e-10;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        double vpara;
        if (norm_k_perp < 1e-6) {
            vpara = 0.0;
        } else {
            vpara = params.waveamp * get_ratio(i, params);
        }
        
        uz_pert += params.amplitude_ratio[i] * vpara * cos(psi); // 示例计算，仅基于 z 和 kx_random
    }
    return uz_pert;
}

double N_PERT(double x, double y, double z, const PerturbationParams& params) {
    double n_pert = 0.0;
    double cs2va2 = params.Cs * params.Cs + params.va * params.va;
    double norm_b0 = sqrt(params.b0x * params.b0x + params.b0y * params.b0y + params.b0z * params.b0z);
    double Vpf;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i] + params.kz_random[i] * params.kz_random[i]);
        double norm_k_perp = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i])+1e-10;
        double cos_kB = (params.kz_random[i] * params.b0z) / norm_k / norm_b0;
        double Vpf = get_param_Vpf(cs2va2, cos_kB, params);
        double v1x, v1y, v1z;
        if (norm_k_perp < 1e-6) {
            v1x = params.waveamp;
            v1y = 0.0;
            v1z = 0.0;
        } else {
            v1x = params.kx_random[i] / norm_k_perp * params.waveamp;
            v1y = params.ky_random[i] / norm_k_perp * params.waveamp;
            v1z = params.waveamp * get_ratio(i, params);
        }
        double omega = norm_k * Vpf;
        double psi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        n_pert += params.amplitude_ratio[i] * params.n0 * (params.kx_random[i] * v1x + params.ky_random[i] * v1y + params.kz_random[i] * v1z) / omega * cos(psi);
    } 
    return n_pert;
}

double EX_PERT(double x, double y, double z, const PerturbationParams& params) {
    double Ex_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k_perp = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i])+1e-10;
        double v1y;
        if (norm_k_perp < 1e-6) {
            v1y = 0.0;
        } else {
            v1y = params.ky_random[i] / norm_k_perp * params.waveamp;
        }
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        Ex_pert += - params.amplitude_ratio[i] * v1y * params.b0z * cos(psi);
    }
    return Ex_pert;
}


double EY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double Ey_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k_perp = sqrt(params.kx_random[i] * params.kx_random[i] + params.ky_random[i] * params.ky_random[i])+1e-10;
        double v1x;
        if (norm_k_perp < 1e-6) {
            v1x = params.waveamp;
        } else {
            v1x = params.kx_random[i] / norm_k_perp * params.waveamp;
        }
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        Ey_pert += params.amplitude_ratio[i] * v1x * params.b0z * cos(psi);
    }
    return Ey_pert;
}

double EZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double Ez_pert = 0.0;
    return Ez_pert;
}


