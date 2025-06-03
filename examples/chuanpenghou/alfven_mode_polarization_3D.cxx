#include <vector>
#include <random>
#include <cmath>
#include <cstdlib> // for rand()
#include <ctime>   // for seeding rand()

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

double get_cos_k_B0(int i, const PerturbationParams& params) {
    std::vector<double> k = {params.kx_random[i], params.ky_random[i], params.kz_random[i]};
    std::vector<double> B0 = {params.b0x, params.b0y, params.b0z};
    double norm_k = vector_norm(k);
    double norm_B0 = vector_norm(B0);
    if (norm_k == 0.0 || norm_B0 == 0.0) {
        return 0.0;
    }
    double dot_product = k[0] * B0[0] + k[1] * B0[1] + k[2] * B0[2];
    return dot_product / (norm_k * norm_B0);
}
// 计算两个向量的叉积
std::vector<double> cross_product(const std::vector<double>& a, const std::vector<double>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

std::vector<double> random_unit_vector(std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0, 2 * M_PI);
    double theta = dist(rng);
    return {cos(theta), sin(theta), 0.0};
}

std::vector<double> get_k_cross_B0(int i, const PerturbationParams& params) {
    std::vector<double> k = {params.kx_random[i], params.ky_random[i], params.kz_random[i]};
    std::vector<double> B0 = {params.b0x, params.b0y, params.b0z};
    std::vector<double> cross = cross_product(k, B0);
    double cross_norm = vector_norm(cross);
    
    if (cross_norm < 1e-6) {
        // 处理叉乘为零的情况：返回默认单位向量,x轴方向
        std::mt19937 rng(i);
	return random_unit_vector(rng);
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
        std::vector<double> e_k_b0 = get_k_cross_B0(i, params);
        double cos_k_B0 = get_cos_k_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1z = Bx * v1y - By * v1x;
        double E1y = Bz * v1x - Bx * v1z;
        double omega = norm_k * params.va * std::abs(cos_k_B0);
        double b1x = (params.ky_random[i] * E1z - params.kz_random[i] * E1y) / omega;
        double phi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if ((std::abs(norm_k)>1e-6) && (std::abs(omega) > 1e-6)) bx_pert += params.amplitude_ratio[i] * b1x * cos(phi);
    }
    return bx_pert;
}

double BY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double by_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] +
                            params.ky_random[i] * params.ky_random[i] +
                            params.kz_random[i] * params.kz_random[i]);
        std::vector<double> e_k_b0 = get_k_cross_B0(i, params);
        double cos_k_B0 = get_cos_k_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1z = Bx * v1y - By * v1x;
        double E1x = By * v1z - Bz * v1y;
        double omega = norm_k * params.va * std::abs(cos_k_B0);
        double b1y = (params.kz_random[i] * E1x - params.kx_random[i] * E1z) / omega;
        double phi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if ((std::abs(norm_k)>1e-6) && (std::abs(omega) > 1e-6)) by_pert += params.amplitude_ratio[i] * b1y * cos(phi);
    }
    return by_pert;
}
double BZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double bz_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        double norm_k = sqrt(params.kx_random[i] * params.kx_random[i] +
                             params.ky_random[i] * params.ky_random[i] +
                             params.kz_random[i] * params.kz_random[i]);
        std::vector<double> e_k_b0 = get_k_cross_B0(i, params);
        double cos_k_B0 = get_cos_k_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1x = By * v1z - Bz * v1y;
        double E1y = Bz * v1x - Bx * v1z;
        double omega = norm_k * params.va * std::abs(cos_k_B0);
        double b1z = (params.kx_random[i] * E1y - params.ky_random[i] * E1x) / omega;
        double phi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if ((std::abs(norm_k)>1e-6) && (std::abs(omega) > 1e-6)) bz_pert += params.amplitude_ratio[i] * b1z * cos(phi);
    }
    return bz_pert;
}

double UX_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ux_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_cross_B0(i, params);
        double cos_k_B0 = get_cos_k_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if (std::abs(cos_k_B0) > 1e-6) ux_pert += params.amplitude_ratio[i] * v1x * cos(psi); 
    }
    return ux_pert;
}

double UY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double uy_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_cross_B0(i, params);
        double cos_k_B0 = get_cos_k_B0(i, params);
        double v1y = e_k_b0[1] * params.waveamp;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if (std::abs(cos_k_B0) > 1e-6) uy_pert += params.amplitude_ratio[i] * v1y * cos(psi);
    }
    return uy_pert;
}

double UZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double uz_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_cross_B0(i, params);
        double cos_k_B0 = get_cos_k_B0(i, params);
        double v1z = e_k_b0[2] * params.waveamp;
        double psi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if (std::abs(cos_k_B0) > 1e-6) uz_pert += params.amplitude_ratio[i] * v1z * cos(psi); 
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
        std::vector<double> e_k_b0 = get_k_cross_B0(i, params);
        double cos_k_B0 = get_cos_k_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1x = By * v1z - Bz * v1y;
        double phi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if (std::abs(cos_k_B0) > 1e-6) ex_pert += params.amplitude_ratio[i] * E1x * cos(phi);
    }
    return ex_pert;
}

double EY_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ey_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_cross_B0(i, params);
        double cos_k_B0 = get_cos_k_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1y = Bz * v1x - Bx * v1z;
        double phi = params.kx_random[i] * x + params.ky_random[i] * y + params.kz_random[i] * z + params.phi_random[i];
        if (std::abs(cos_k_B0) > 1e-6) ey_pert += params.amplitude_ratio[i] * E1y * cos(phi);
    }
    return ey_pert;
}

double EZ_PERT(double x, double y, double z, const PerturbationParams& params) {
    double ez_pert = 0.0;
    for (int i = 0; i < params.n_modes; i++) {
        std::vector<double> e_k_b0 = get_k_cross_B0(i, params);
        double cos_k_B0 = get_cos_k_B0(i, params);
        double v1x = e_k_b0[0] * params.waveamp;
        double v1y = e_k_b0[1] * params.waveamp;
        double v1z = e_k_b0[2] * params.waveamp;
        double Bx = params.b0x;
        double By = params.b0y;
        double Bz = params.b0z;
        double E1z = Bx * v1y - By * v1x;
        double phi = params.kx_random[i]*x + params.ky_random[i]*y + params.kz_random[i]*z + params.phi_random[i];
        if (std::abs(cos_k_B0) > 1e-6) ez_pert += params.amplitude_ratio[i] * E1z * cos(phi);
    }
    return ez_pert;
}
