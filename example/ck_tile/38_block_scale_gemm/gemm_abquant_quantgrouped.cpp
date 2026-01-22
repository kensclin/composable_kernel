// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "run_gemm_quant_example.inc"

//template <typename T>
//using GemmConfig = GemmConfigQuantPrefill<T>;
template <typename T>
struct GemmConfigAsync : public GemmConfigQuantPrefill<T>
{
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Tile = 192;
    static constexpr ck_tile::index_t N_Tile = 128 * N_Warp;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(T) * K_Warp;

    static constexpr bool kPadK      = false;
    static constexpr bool TransposeC = true;
    static constexpr int kBlockPerCu = 1;

};
template <typename T>
struct GemmConfigPreshuffleB : public GemmConfigPreshuffleB_BQuant_Prefill<T>
{
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr bool kPadK      = false;
    static constexpr bool TransposeC = true;
};

template <typename T>
struct GemmConfig : public GemmConfigQuantPrefill<T>
{
    static constexpr bool kPadK      = false;
    static constexpr bool TransposeC = true;
};

void abquant_quantgrouped_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut)
{
    lut[hash_multiple_strings({"fp8",
                               "abquant",
                               "non-preshuffleb",
                               "non-preshufflequant",
                               "1x1x128"})] = [](const ck_tile::ArgParser& arg_parser) {
        using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          ck_tile::QuantType::ABQuantGrouped>(arg_parser);
    };
    lut[hash_multiple_strings({"fp8",
                               "abquant",
                               "non-preshuffleb",
                               "non-preshufflequant",
                               "1x128x128"})] = [](const ck_tile::ArgParser& arg_parser) {
        using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfigAsync<ck_tile::fp8_t>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          ck_tile::QuantType::ABQuantGrouped>(arg_parser);
    };
    lut[hash_multiple_strings({"bf8",
                               "abquant",
                               "non-preshuffleb",
                               "non-preshufflequant",
                               "1x1x128"})] = [](const ck_tile::ArgParser& arg_parser) {
        using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          ck_tile::QuantType::ABQuantGrouped>(arg_parser);
    };
    lut[hash_multiple_strings({"bf8",
                               "abquant",
                               "non-preshuffleb",
                               "non-preshufflequant",
                               "1x128x128"})] = [](const ck_tile::ArgParser& arg_parser) {
        using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfigAsync<ck_tile::bf8_t>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          ck_tile::QuantType::ABQuantGrouped>(arg_parser);
    };
    lut[hash_multiple_strings({"fp8",
                               "abquant",
                               "preshuffleb",
                               "non-preshufflequant",
                               "1x1x128"})] = [](const ck_tile::ArgParser& arg_parser) {
        using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfigPreshuffleB<ck_tile::fp8_t>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          ck_tile::QuantType::ABQuantGrouped>(arg_parser);
    };
    lut[hash_multiple_strings({"fp8",
                               "abquant",
                               "preshuffleb",
                               "non-preshufflequant",
                               "1x128x128"})] = [](const ck_tile::ArgParser& arg_parser) {
        using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfigPreshuffleB<ck_tile::fp8_t>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          ck_tile::QuantType::ABQuantGrouped>(arg_parser);
    };
    lut[hash_multiple_strings({"bf8",
                               "abquant",
                               "preshuffleb",
                               "non-preshufflequant",
                               "1x1x128"})] = [](const ck_tile::ArgParser& arg_parser) {
        using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfigPreshuffleB<ck_tile::bf8_t>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          ck_tile::QuantType::ABQuantGrouped>(arg_parser);
    };
    lut[hash_multiple_strings({"bf8",
                               "abquant",
                               "preshuffleb",
                               "non-preshufflequant",
                               "1x128x128"})] = [](const ck_tile::ArgParser& arg_parser) {
        using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfigPreshuffleB<ck_tile::bf8_t>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          ck_tile::QuantType::ABQuantGrouped>(arg_parser);
    };
}
