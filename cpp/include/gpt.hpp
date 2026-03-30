#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

// Weights for a single transformer block
struct BlockWeights
{
    // Layer Normalization 1
    Eigen::VectorXf ln1_weight;
    Eigen::VectorXf ln1_bias;

    // Fused QKV Projection
    Eigen::MatrixXf c_attn_weight; //[2304, 768]
    Eigen::VectorXf c_attn_bias; //[2304]

    Eigen::MatrixXf c_proj_weight; //[768, 768]
    Eigen::VectorXf c_proj_bias; //[768]



    // Layer Normalization 2
    Eigen::VectorXf ln2_weight;
    Eigen::VectorXf ln2_bias;

    // FFN first Linear Layer
    Eigen::MatrixXf mlp_fc_weight; // [3072, 768]
    Eigen::VectorXf mlp_fc_bias; // [3072]

    // FFN second Linear Layer
    Eigen::MatrixXf mlp_proj_weight; // [768, 3072]
    Eigen::VectorXf mlp_proj_bias; // [768]
};

class GPT2Inference
{
    private :
        Eigen::MatrixXf wte; // [50257, 768] - Token Embedding
        Eigen::MatrixXf wpe; // [1024, 768] - Positional Embeddings

        // 12 Transformer blocks
        std::vector<BlockWeights> blocks;

        //Final Layer Normalization
        Eigen::VectorXf ln_f_weight;
        Eigen::VectorXf ln_f_bias;

        // Configuration
        const int n_layer = 12;
        const int n_head = 12;
        const int n_embd = 768;
        const int d_head = 64; // n_embd/n_head

        void load_weights(const std::string& path);

        Eigen::MatrixXf layer_norm(const Eigen::MatrixXf& x,
            const Eigen::VectorXf& gamma, const Eigen::VectorXf& beta);

        Eigen::MatrixXf gelu(const Eigen::MatrixXf& x);

        Eigen::MatrixXf ffn(const Eigen::MatrixXf& x, const BlockWeights& bw);

        Eigen::MatrixXf attention(const Eigen::MatrixXf& x, const BlockWeights& bw);

        Eigen::MatrixXf transformer_block(const Eigen::MatrixXf& x, const BlockWeights& bw);

    public :
        GPT2Inference(const std::string& bin_file_path)
        {
            load_weights(bin_file_path);
        }

        std::vector<float> forward_pass(const std::vector<int>& tokens);
};
