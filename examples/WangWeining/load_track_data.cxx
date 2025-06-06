#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>
#ifdef __cpp_lib_filesystem
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

// 粒子数据结构
struct Particle {
    int id;          // 粒子 ID
    double x, y, z;  // 绝对位置
    double vx, vy, vz; // 速度
};

// 临时结构体，用于读取二进制数据
struct RawParticleData {
    float dxyz[3];
    int32_t icell;
    float u[3];
    float q;
    int32_t id;
};

// === 新增：筛选功能（通过宏控制是否启用）===
#ifdef ENABLE_FILTER
// 位置范围结构体
struct PositionRange {
    double x_min, x_max;
    double y_min, y_max;
    double z_min, z_max;
    PositionRange(double x1=0, double x2=0, double y1=0, double y2=0, double z1=0, double z2=0)
        : x_min(x1), x_max(x2), y_min(y1), y_max(y2), z_min(z1), z_max(z2) {}
};
struct VelocityRange {
    double vx_min, vx_max;
    double vy_min, vy_max;
    double vz_min, vz_max;
    VelocityRange(double vx1=0, double vx2=0, 
                  double vy1=0, double vy2=0, 
                  double vz1=0, double vz2=0)
        : vx_min(vx1), vx_max(vx2),
          vy_min(vy1), vy_max(vy2),
          vz_min(vz1), vz_max(vz2) {}
};
// 能量范围结构体（基于动能 E=0.5*v²，假设质量m=1）
struct EnergyRange {
    double min_energy, max_energy;
    EnergyRange(double min_e=0, double max_e=0)
        : min_energy(min_e), max_energy(max_e) {}
};
struct TimeRange {
    int start_timestep;  // 起始时间步（物理时间，如0, 10, 20...）
    int end_timestep;    // 结束时间步（包含）
    TimeRange(int start = -1, int end = -1) 
        : start_timestep(start), end_timestep(end) {}
};
#endif

class ParticleTrajectoryTracker {
private:
    std::vector<std::vector<Particle>> timestepData;
    std::vector<std::unordered_map<int, size_t>> idToIndexMaps;
    std::vector<int> particleIds;
    int topoX, topoY, topoZ;
    std::string baseDir;
    std::string particleFilePattern;
    int particleOutputInterval;

public:
    ParticleTrajectoryTracker(int topoX, int topoY, int topoZ, 
                             const std::string& baseDir,
                             int particleOutputInterval = 1,
                             const std::string& particleFilePattern = "particle.")
        : topoX(topoX), topoY(topoY), topoZ(topoZ), 
          baseDir(baseDir), 
          particleOutputInterval(particleOutputInterval),
          particleFilePattern(particleFilePattern) {}

    void loadAllTimesteps(int startTimestep, int endTimestep) {
        for (int t = startTimestep; t <= endTimestep; t += particleOutputInterval) {
            loadTimestepData(t);
            std::cout << "Loaded timestep " << t << std::endl;
        }
    }

    void loadTimestepData(int timestep) {
        std::string timestepDir = baseDir + "/T." + std::to_string(timestep);
        if (!fs::exists(timestepDir)) {
            throw std::runtime_error("时间步目录不存在: " + timestepDir);
        }

        std::vector<Particle> allParticles;
        std::unordered_map<int, size_t> idMap;
        int totalProcesses = topoX * topoY * topoZ;

        for (int rank = 0; rank < totalProcesses; ++rank) {
            std::string filename = timestepDir + "/" + particleFilePattern + 
                                  std::to_string(timestep) + "." + std::to_string(rank);
            if (fs::exists(filename)) {
                std::vector<Particle> particles = readParticleFile(filename);
                size_t startIdx = allParticles.size();
                allParticles.insert(allParticles.end(), particles.begin(), particles.end());
                for (size_t i = 0; i < particles.size(); ++i) {
                    idMap[particles[i].id] = startIdx + i;
                }
            } else {
                std::cerr << "警告: 文件不存在 " << filename << std::endl;
            }
        }

        // 更新全局粒子ID列表（去重）
        std::unordered_set<int32_t> idSet;
        particleIds.clear();
        for (const auto& p : allParticles) {
            if (idSet.insert(p.id).second) {
                particleIds.push_back(p.id);
            }
        }

        timestepData.push_back(std::move(allParticles));
        idToIndexMaps.push_back(std::move(idMap));
    }

    std::vector<Particle> readParticleFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开文件: " + filename);
        }

        // 读取样板数据（前23字节）
        char boilerplate[23];
        file.read(boilerplate, sizeof(boilerplate));

        // 读取头部参数
        int32_t version, type, nt, nx, ny, nz;
        float dt, dx, dy, dz, x0, y0, z0, cvac, eps0, damp;
        int32_t rank, ndom, spid, spqm;

        auto read_int32 = [&](int32_t& val) {
            file.read(reinterpret_cast<char*>(&val), sizeof(int32_t));
        };
        auto read_float = [&](float& val) {
            file.read(reinterpret_cast<char*>(&val), sizeof(float));
        };

        read_int32(version); read_int32(type); read_int32(nt);
        read_int32(nx); read_int32(ny); read_int32(nz);
        read_float(dt); read_float(dx); read_float(dy);
        read_float(dz); read_float(x0); read_float(y0);
        read_float(z0); read_float(cvac); read_float(eps0);
        read_float(damp); read_int32(rank); read_int32(ndom);
        read_int32(spid); read_int32(spqm);

        // 读取粒子数量
        int32_t size, ndim, dim;
        read_int32(size); read_int32(ndim); read_int32(dim);
        nx += 2; ny += 2;
        int particleCount = dim;

        std::vector<RawParticleData> rawParticles(particleCount);
        file.read(reinterpret_cast<char*>(rawParticles.data()), 
                  particleCount * sizeof(RawParticleData));
        if (file.gcount() != particleCount * sizeof(RawParticleData)) {
            std::cerr << "警告: 读取字节数不匹配" << std::endl;
        }

        // 转换为物理坐标和速度
        std::vector<Particle> particles(particleCount);
        for (int i = 0; i < particleCount; ++i) {
            const auto& raw = rawParticles[i];
            particles[i].id = raw.id;

            // 计算网格坐标
            int ix = raw.icell % nx;
            int iy = (raw.icell / nx) % ny;
            int iz = raw.icell / (nx * ny);

            // 绝对位置
            particles[i].x = x0 + (ix - 1.0f + (raw.dxyz[0] + 1.0f) * 0.5f) * dx;
            particles[i].y = y0 + (iy - 1.0f + (raw.dxyz[1] + 1.0f) * 0.5f) * dy;
            particles[i].z = z0 + (iz - 1.0f + (raw.dxyz[2] + 1.0f) * 0.5f) * dz;

            // 计算速度（假设相对论速度，gamma=1/sqrt(1-u²/c²)，此处c=1）
            float gamma = std::sqrt(1.0f + raw.u[0]*raw.u[0] + raw.u[1]*raw.u[1] + raw.u[2]*raw.u[2]);
            particles[i].vx = raw.u[0] / gamma;
            particles[i].vy = raw.u[1] / gamma;
            particles[i].vz = raw.u[2] / gamma;
        }
        return particles;
    }

    // === 原始功能：按ID获取轨迹 ===
    std::vector<std::vector<Particle>> getTrajectories(const std::vector<int>& selectedIds) const {
        std::vector<std::vector<Particle>> trajectories(selectedIds.size());
        std::cout << "生成 " << selectedIds.size() << " 个粒子的轨迹" << std::endl;
        for (size_t i = 0; i < selectedIds.size(); ++i) {
            int particleId = selectedIds[i];
            std::vector<Particle>& traj = trajectories[i];
            std::cout << "  粒子ID: " << particleId << std::endl;
            for (size_t t = 0; t < timestepData.size(); ++t) {
                auto it = idToIndexMaps[t].find(particleId);
                if (it != idToIndexMaps[t].end()) {
                    traj.push_back(timestepData[t][it->second]);
                    std::cout << "    在时间步 " << t << " 找到粒子" << std::endl;
                } else {
                    traj.push_back({particleId, 0, 0, 0, 0, 0, 0}); // 标记缺失
                    std::cout << "    在时间步 " << t << " 未找到粒子" << std::endl;
                }
            }
        }
        return trajectories;
    }

    // === 新增功能：按条件筛选轨迹（需定义ENABLE_FILTER宏）===
#ifdef ENABLE_FILTER
// 筛选并追踪函数
std::vector<std::vector<Particle>> filterAndTrackParticles(
    int reference_timestep,          // 参考时间步（用于筛选条件）
    const PositionRange& pos_range,  // 位置范围
    const EnergyRange& energy_range,  // 能量范围
    const VelocityRange& vel_range  // 速度范围
) const {
    std::vector<std::vector<Particle>> trajectories;
    std::vector<int> selected_particle_ids;  // 存储在参考时间步符合条件的粒子ID

    // 1. 在参考时间步中筛选符合条件的粒子
    size_t ref_timestep_idx = reference_timestep / particleOutputInterval;
    if (ref_timestep_idx >= timestepData.size()) {
        throw std::out_of_range("参考时间步超出范围");
    }

    const auto& ref_id_map = idToIndexMaps[ref_timestep_idx];
    const auto& ref_particles = timestepData[ref_timestep_idx];

    for (const auto& pair : ref_id_map) {
        const Particle& p = ref_particles[pair.second];
        bool within_pos = true, within_energy = true, within_velocity = true;

        // 位置筛选
        if (pos_range.x_min <= pos_range.x_max) {
            within_pos = (p.x >= pos_range.x_min && p.x <= pos_range.x_max &&
                          p.y >= pos_range.y_min && p.y <= pos_range.y_max &&
                          p.z >= pos_range.z_min && p.z <= pos_range.z_max);
        }

        // 能量筛选
        double velocity_sq = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
        double energy = 0.5 * velocity_sq;
        if (energy_range.min_energy <= energy_range.max_energy) {
            within_energy = (energy >= energy_range.min_energy && 
                            energy <= energy_range.max_energy);
        }
        // 速度筛选（可选，根据需要启用）
         if (!std::isnan(vel_range.vx_min)) { // 非NaN表示启用该维度筛选
            within_velocity = (p.vx >= vel_range.vx_min && p.vx <= vel_range.vx_max &&
                             p.vy >= vel_range.vy_min && p.vy <= vel_range.vy_max &&
                             p.vz >= vel_range.vz_min && p.vz <= vel_range.vz_max);
        }

        if (within_pos && within_energy) {
            selected_particle_ids.push_back(p.id);
        }
    }

    std::cout << "在时间步 " << reference_timestep 
              << " 筛选出 " << selected_particle_ids.size() << " 个粒子" << std::endl;

    // 2. 追踪这些粒子在所有时间步的轨迹
    for (int pid : selected_particle_ids) {
        std::vector<Particle> trajectory;
        bool found_in_any_step = false;

        for (size_t t = 0; t < timestepData.size(); ++t) {
            const auto& id_map = idToIndexMaps[t];
            const auto& particles = timestepData[t];
            auto it = id_map.find(pid);

            if (it != id_map.end()) {
                // 粒子存在，添加到轨迹（无论是否仍在筛选范围内）
                trajectory.push_back(particles[it->second]);
                found_in_any_step = true;
            } else {
                // 粒子不存在，添加占位符
                trajectory.push_back({pid, 0, 0, 0, 0, 0, 0});
            }
        }

        if (found_in_any_step) {
            trajectories.push_back(trajectory);
        }
    }

    return trajectories;
}
#endif

    const std::vector<int>& getAllParticleIds() const { return particleIds; }

    void saveTrajectories(const std::vector<std::vector<Particle>>& trajectories, 
                         const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开文件: " + filename);
        }

        file << trajectories.size() << std::endl;
        for (const auto& traj : trajectories) {
            file << traj[0].id << " " << traj.size() << std::endl;
            for (const auto& p : traj) {
                file << p.x << " " << p.y << " " << p.z << " "
                     << p.vx << " " << p.vy << " " << p.vz << std::endl;
            }
        }
        file.close();
    }
};

// === 使用示例 ===
int main() {
    try {
        // 初始化追踪器（原始参数示例）
        ParticleTrajectoryTracker tracker(
            8, 1, 2,          // MPI拓扑
            "particle",       // 数据根目录
            10,               // 输出间隔
            "Hparticle_c."    // 文件模式
        );

        // 加载时间步0到50（间隔10，实际加载0,10,20,30,40,50）
        tracker.loadAllTimesteps(0, 50);

        // === 原始功能：按ID获取轨迹 ===
        const auto& all_ids = tracker.getAllParticleIds();
        std::vector<int> selected_ids;
        if (!all_ids.empty()) {
            // 选取前1000个粒子
            selected_ids.assign(all_ids.begin(), 
                               all_ids.begin() + std::min(1000, static_cast<int>(all_ids.size())));
        }
        std::cout << "选取 " << selected_ids.size() << " 个粒子进行追踪" << std::endl;
        auto trajectories = tracker.getTrajectories(selected_ids);
        tracker.saveTrajectories(trajectories, "particle_trajectories.txt");
        std::cout << "原始轨迹已保存" << std::endl;

        // === 新增功能：按条件筛选（需启用ENABLE_FILTER宏）===
    #ifdef ENABLE_FILTER
    // 筛选时间步t=10中，位于[0,5]^3且能量在[0.1,0.5]的粒子，并追踪它们的完整轨迹
    int reference_timestep = 10;  // 物理时间步
    PositionRange pos(0, 5, 0, 5, 0, 5);
    EnergyRange energy(0.1, 0.5);

    auto trajectories_2 = tracker.filterAndTrackParticles(
        reference_timestep,
        pos,
        energy
    );

    // 保存轨迹（包含所有时间步的数据）
    tracker.saveTrajectories(trajectories_2, "tracked_particles.txt");
    #endif

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}