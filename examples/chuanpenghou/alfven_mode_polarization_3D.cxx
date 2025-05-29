
struct PerturbationParams {
    std::vector<double> amplitude_ratio;
    std::vector<double> kx_random;
    std::vector<double> ky_random;
    std::vector<double> kz_random;
    std::vector<double> phi_random;
    int n_modes;
    double n0;
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
std::vector<double> get_k_corss_B0(int i, const PerturbationParams& params) {
    std::vector<double> k = {params.kx_random[i], params.ky_random[i], params.kz_random[i]};
    std::vector<double> B0 = {params.b0x, params.b0y, params.b0z};
    std::vector<double> cross = cross_product(k, B0);
    double cross_norm = vector_norm(cross);
    
    if (cross_norm == 0.0) {
        // 处理叉积为零的情况：返回默认单位向量（如z轴方向）或根据业务调整
        return {0.0, 0.0, 1.0}; // 示例默认方向，可改为抛出异常或其他处理方式
    }
    
    std::vector<double> e_k_b0 = {
        cross[0] / cross_norm,
        cross[1] / cross_norm,
        cross[2] / cross_norm
    };
    return e_k_b0;
}

double BX_PERT(double x, double y, double z, const PerturbationParams& params) {
    double bx_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] +
                             params.ky_random[i] * params.ky_random[i] +
                             params.kz_random[i] * params.kz_random[i]);
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1z = Bx * v1y - By * v1x;
        double E1y = Bz * v1x - Bx * v1z;
        double b1x = params.ky_random[i] * E1z - params.kz_random[i] * E1y;
        double phi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if (std::fabs(norm_k)>1e-6)bx_pert += params.amplitude_ratio[i] * b1x / norm_k / params.va * cos(phi);
    }
    return bx_pert;
}

double BY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double by_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] +
                            params.ky_random[i] * params.ky_random[i] +
                            params.kz_random[i] * params.kz_random[i]);
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1z = Bx * v1y - By * v1x;
        double E1x = By * v1z - Bz * v1y;
        double b1y = params.kz_random[i] * E1x - params.kx_random[i] * E1z;
        double phi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if (std::fabs(norm_k)>1e-6)by_pert += params.amplitude_ratio[i] * b1y  / norm_k / params.va * cos(phi);
    }
    return by_pert;
}
double BZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double bz_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] +
                             params.ky_random[i] * params.ky_random[i] +
                             params.kz_random[i] * params.kz_random[i]);
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1x = By * v1z - Bz * v1y;
        double E1y = Bz * v1x - Bx * v1z;
        double b1z = params.kx_random[i] * E1y - params.ky_random[i] * E1x;
        double phi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if (std::fabs(norm_k)>1e-6){
        bz_pert += params.amplitude_ratio[i] * b1z  / norm_k / params.va * cos(phi);
        }
    }
    return bz_pert;
}

double UX_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ux_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        ux_pert += params.amplitude_ratio[i] * v1x * cos(psi); 
    }
    return ux_pert;
}

double UY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double uy_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1y = e_k_b0[1] * params.waveamp;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        uy_pert += params.amplitude_ratio[i] * v1y * cos(psi);
    }
    return uy_pert;
}

double UZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double uz_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1z = e_k_b0[2] * params.waveamp;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        uz_pert += params.amplitude_ratio[i] * v1z * cos(psi); 
    }
    return uz_pert;
}

double N_PERT(double x, double y, double z, const PerturbationParams& params) {
    double n_pert = 0.0;
    return n_pert;
}
double EX_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ex_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1x = By * v1z - Bz * v1y;
        double phi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        ex_pert += params.amplitude_ratio[i] * E1x * cos(phi);
    }
    return ex_pert;
}

double EY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ey_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1y = Bz * v1x - Bx * v1z;
        double phi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        ey_pert += params.amplitude_ratio[i] * E1y * cos(phi);
    }
    return ey_pert;
}

double EZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ez_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_corss_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1z = Bx * v1y - By * v1x;
        double phi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        ez_pert += params.amplitude_ratio[i] * E1z * cos(phi);
    }
    return ez_pert;
}