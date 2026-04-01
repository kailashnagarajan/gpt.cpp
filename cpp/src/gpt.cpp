#include "gpt.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <stdexcept>
#include <chrono>

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

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c_attn_weight_map(ptr_, 768, 2304);
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

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mlp_fc_weight_map(ptr_, 768, 3072);
        bw.mlp_fc_weight = mlp_fc_weight_map;
        ptr_+= 3072 * 768;

        Eigen::Map<Eigen::VectorXf> vec_map_mlp_fc_bias(ptr_, 3072);
        bw.mlp_fc_bias = vec_map_mlp_fc_bias;
        ptr_+= 3072;

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mlp_proj_weight_map(ptr_,3072,768);
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

Eigen::MatrixXf GPT2Inference::gelu(const Eigen::MatrixXf& x)
{
    auto arr = x.array();
    return (0.5f * arr * (1.0f + ((float)sqrt(2.0/M_PI) * (arr + 0.044715f * arr.cube())).tanh())).matrix();
}

Eigen::MatrixXf GPT2Inference::layer_norm(const Eigen::MatrixXf& x,
    const Eigen::VectorXf& gamma, const Eigen::VectorXf& beta)
{
    float epsilon = 1e-5f;
    // std::vector<Eigen::VectorXf> columns;
    Eigen::VectorXf mean = x.rowwise().mean();
    Eigen::MatrixXf deviation = x.colwise() - mean;
    Eigen::VectorXf variance = deviation.array().square().rowwise().mean();

    Eigen::MatrixXf normalized = (deviation.array().colwise() / (variance.array() + epsilon).sqrt()).matrix();

    Eigen::MatrixXf result = normalized;
    result = (result.array().rowwise() * gamma.transpose().array()).rowwise() + beta.transpose().array();
    return result;
}

Eigen::MatrixXf GPT2Inference::ffn(const Eigen::MatrixXf& x, const BlockWeights& bw)
{
    Eigen::MatrixXf flp = (x * bw.mlp_fc_weight).rowwise() + bw.mlp_fc_bias.transpose();
    Eigen::MatrixXf activated_flp = gelu(flp);
    Eigen::MatrixXf slp = (activated_flp * bw.mlp_proj_weight).rowwise() + bw.mlp_proj_bias.transpose();

    return slp;
}

Eigen::MatrixXf GPT2Inference::attention(const Eigen::MatrixXf& x, const BlockWeights& bw)
{
    Eigen::MatrixXf qkv_matrix = (x * bw.c_attn_weight).rowwise() + bw.c_attn_bias.transpose();

    Eigen::MatrixXf Q = qkv_matrix.block(0, 0,      x.rows(), 768);
    Eigen::MatrixXf K = qkv_matrix.block(0, 768,    x.rows(), 768);
    Eigen::MatrixXf V = qkv_matrix.block(0, 768*2,  x.rows(), 768);

    Eigen::MatrixXf Z = Eigen::MatrixXf::Zero(x.rows(), n_embd);

    for (int i = 0; i<n_head; ++i)
    {
        Eigen::MatrixXf Q_head = Q.block(0, i * 64, x.rows(), 64);
        Eigen::MatrixXf K_head = K.block(0, i * 64, x.rows(), 64);
        Eigen::MatrixXf V_head = V.block(0, i * 64, x.rows(), 64);

        Eigen::MatrixXf attention_score_head = (Q_head * K_head.transpose())/8;
        attention_score_head.triangularView<Eigen::StrictlyUpper>().fill(
            -std::numeric_limits<float>::infinity()
        );

        for (int t = 0; t < x.rows(); ++t)
        {
            Eigen::VectorXf row = attention_score_head.row(t);
            float max_val = row.maxCoeff();
            Eigen::VectorXf exp_row = (row.array() - max_val).exp();
            attention_score_head.row(t) = exp_row / exp_row.sum();
        }

        Eigen::MatrixXf Z_head = attention_score_head * V_head;
        Z.block(0, i * d_head, x.rows(), d_head) = Z_head;
    }

    Eigen::MatrixXf output_projection = (Z * bw.c_proj_weight).rowwise() + bw.c_proj_bias.transpose();

    return output_projection;
}

Eigen::MatrixXf GPT2Inference::transformer_block(const Eigen::MatrixXf& x, const BlockWeights& bw)
{
    Eigen::MatrixXf out_copy = x;
    out_copy = out_copy + attention(layer_norm(out_copy, bw.ln1_weight, bw.ln1_bias), bw);
    out_copy = out_copy + ffn(layer_norm(out_copy, bw.ln2_weight, bw.ln2_bias), bw);

    return out_copy;
}

std::vector<float> GPT2Inference::forward_pass(const std::vector<int>& tokens)
{
    int seq_len = tokens.size();
    Eigen::MatrixXf x(seq_len, n_embd);

    for (int i = 0; i < seq_len; i++)
    {
        x.row(i) = wte.col(tokens[i]) + wpe.col(i);
    }

    for (int i = 0; i < n_layer; i++)
    {
        x = transformer_block(x, blocks[i]);
    }

    x = layer_norm(x, ln_f_weight, ln_f_bias);

    Eigen::MatrixXf logits = x * wte;

    Eigen::VectorXf last_row = logits.row(logits.rows() - 1);

    return std::vector<float>(last_row.data(), last_row.data() + last_row.size());
}

std::pair<std::vector<float>, std::vector<double>> GPT2Inference::forward_timed(const std::vector<int>& tokens)
{

    std::vector<double> timings;
    int seq_len = tokens.size();
    Eigen::MatrixXf x(seq_len, n_embd);

    auto start_x_row = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < seq_len; i++)
    {
        x.row(i) = wte.col(tokens[i]) + wpe.col(i);
    }
    auto end_x_row = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end_x_row - start_x_row).count();
    timings.push_back(ms);

    for (int i = 0; i < n_layer; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        x = transformer_block(x, blocks[i]);
        auto end = std::chrono::high_resolution_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    auto start_ln = std::chrono::high_resolution_clock::now();

    x = layer_norm(x, ln_f_weight, ln_f_bias);

    auto end_ln = std::chrono::high_resolution_clock::now();
    double ms_ln = std::chrono::duration<double, std::milli>(end_ln - start_ln).count();
    timings.push_back(ms_ln);


    Eigen::MatrixXf logits = x * wte;

    Eigen::VectorXf last_row = logits.row(logits.rows() - 1);

    return {std::vector<float>(last_row.data(), last_row.data() + last_row.size()), timings};
}
