#include "gpt.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <stdexcept>

void GPT2Inference::load_weights(const std::string& path)
{
    int fd = open(path.c_str(), O_RDONLY);

    if(fd < 0) throw std::runtime_error("Failed to open weights file: " + path);
    struct stat sb;
    fstat(fd, &sb);

    void* mapped = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if(mapped == MAP_FAILED) throw std::runtime_error("mmap failed");

    close(fd);

    std::int32_t* header = reinterpret_cast<std::int32_t*>(mapped);
    assert(header[0] == 20240520 && "Wrong Magic Number");

    float* ptr_ = reinterpret_cast<float*> (header + 256);

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> wte_map(ptr_, 50257, 768);
    wte = wte_map.transpose();
    ptr_ += 50257 * 768;

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> wpe_map(ptr_, 1024, 768);
    wpe = wpe_map.transpose();
    ptr_ += 1024 * 768;

    for(int i = 0; i<n_layer; ++i)
    {
        BlockWeights bw;
        Eigen::Map<Eigen::VectorXf> vec_map_ln1_weights(ptr_, 768);
        bw.ln1_weight = vec_map_ln1_weights;
        ptr_ += 768;

        Eigen::Map<Eigen::VectorXf> vec_map_ln1_bias(ptr_, 768);
        bw.ln1_bias = vec_map_ln1_bias;
        ptr_+=768;

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c_attn_weight_map(ptr_, 2304, 768);
        bw.c_attn_weight = c_attn_weight_map;
        ptr_+= 2304 * 768;

        Eigen::Map<Eigen::VectorXf> vec_map_c_attn_bias(ptr_, 2304);
        bw.c_attn_bias = vec_map_c_attn_bias;
        ptr_+= 2304;

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c_proj_weight_map(ptr_, 768, 768);
        bw.c_proj_weight = c_proj_weight_map;
        ptr_+= 768 * 768;

        Eigen::Map<Eigen::VectorXf> vec_map_c_proj_bias(ptr_, 768);
        bw.c_proj_bias = vec_map_c_proj_bias;
        ptr_+= 768;

        Eigen::Map<Eigen::VectorXf> vec_map_ln2_weights(ptr_, 768);
        bw.ln2_weight = vec_map_ln2_weights;
        ptr_+= 768;

        Eigen::Map<Eigen::VectorXf> vec_map_ln2_bias(ptr_, 768);
        bw.ln2_bias = vec_map_ln2_bias;
        ptr_+= 768;

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mlp_fc_weight_map(ptr_, 3072, 768);
        bw.mlp_fc_weight = mlp_fc_weight_map;
        ptr_+= 3072 * 768;

        Eigen::Map<Eigen::VectorXf> vec_map_mlp_fc_bias(ptr_, 3072);
        bw.mlp_fc_bias = vec_map_mlp_fc_bias;
        ptr_+= 3072;

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mlp_proj_weight_map(ptr_, 3072, 768);
        bw.mlp_proj_weight = mlp_proj_weight_map;
        ptr_+= 768 * 3072;

        Eigen::Map<Eigen::VectorXf> vec_map_mlp_proj_bias(ptr_, 768);
        bw.mlp_proj_bias = vec_map_mlp_proj_bias;
        ptr_+= 768;

        blocks.push_back(bw);
    }

    Eigen::Map<Eigen::VectorXf> vec_map_ln_f_weight(ptr_, 768);
    ln_f_weight = vec_map_ln_f_weight;
    ptr_+= 768;

    Eigen::Map<Eigen::VectorXf> vec_map_ln_f_bias(ptr_, 768);
    ln_f_bias = vec_map_ln_f_bias;
}
