import MLX
import MLXNN

enum Qwen38DFlash2ConvKernel {
    private static let rows = 8
    private static let hiddenSize = 5_120
    private static let threadsPerThreadgroup = 256

    private static let kernel = MLXFast.metalKernel(
        name: "qwen38_dflash2_dynamic_conv_m8",
        inputNames: ["input", "dynamic", "base", "residual"],
        outputNames: ["out"],
        source: #"""
            constexpr uint Hidden = 5120;
            constexpr uint ConvGroups = 320;
            constexpr uint GroupSize = 16;

            const uint element = thread_position_in_grid.x;
            const uint row = element / Hidden;
            const uint channel = element % Hidden;
            const uint group = channel / GroupSize;
            const uint kind = FUSE_RESIDUAL ? 1 : 0;
            bfloat value = bfloat(0.0f);

            #pragma unroll
            for (uint offset = 0; offset < 2; ++offset) {
                const bfloat source = row >= offset
                    ? bfloat(input[(row - offset) * Hidden + channel])
                    : bfloat(0.0f);
                const bfloat fixedProduct = bfloat(
                    bfloat(base[(kind * 2 + offset) * Hidden + channel])
                    * source);
                const bfloat dynamicProduct = bfloat(
                    bfloat(dynamic[
                        row * 4 * ConvGroups
                        + (kind * 2 + offset) * ConvGroups + group])
                    * source);
                value = bfloat(value + fixedProduct);
                value = bfloat(value + dynamicProduct);
            }
            if (FUSE_RESIDUAL) {
                value = bfloat(bfloat(residual[element]) + value);
            }
            out[element] = value;
            """#,
        ensureRowContiguous: false
    )

    static func call(
        _ input: MLXArray,
        dynamic: MLXArray,
        base: MLXArray,
        residual: MLXArray,
        fuseResidual: Bool
    ) -> MLXArray {
        kernel(
            [input, dynamic, base, residual],
            template: [("FUSE_RESIDUAL", fuseResidual)],
            grid: (rows * hiddenSize, 1, 1),
            threadGroup: (threadsPerThreadgroup, 1, 1),
            outputShapes: [[1, rows, hiddenSize]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}

enum Qwen38DFlash2AttentionKernel {
    private static let attentionHeads = 32
    private static let rows = 8
    private static let headDimension = 128
    private static let threadsPerThreadgroup = 1_024

    private static let kernel = MLXFast.metalKernel(
        name: "qwen38_dflash2_attention_m8_d128",
        inputNames: [
            "queries",
            "ring_keys", "ring_values",
            "context_keys", "context_values",
            "proposal_keys", "proposal_values",
            "cache_state",
        ],
        outputNames: ["out"],
        source: #"""
            constexpr uint Rows = 8;
            constexpr uint HeadDimension = 128;
            constexpr uint Partitions = 32;
            constexpr uint ValuesPerLane = 4;
            constexpr uint HeadsPerKV = 4;
            constexpr uint KVHeads = 8;
            constexpr uint QueryHeads = 32;
            constexpr uint Threads = 1024;
            constexpr uint Window = 2048;
            constexpr float Scale = 0.08838834765f;

            const uint queryHead = threadgroup_position_in_grid.x;
            const uint kvHead = queryHead / HeadsPerKV;
            const uint partition = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint retainedLength = uint(cache_state[0]);
            const uint contextStart = uint(cache_state[1]);
            const uint contextLength = uint(context_keys_shape[2]);
            const uint cachedLength = retainedLength + contextLength;
            const uint totalLength = cachedLength + Rows;

            device bfloat* writableKeys =
                const_cast<device bfloat*>(ring_keys);
            device bfloat* writableValues =
                const_cast<device bfloat*>(ring_values);
            for (uint element = thread_position_in_grid.x;
                 element < KVHeads * contextLength * HeadDimension;
                 element += QueryHeads * Threads) {
                const uint dimension = element % HeadDimension;
                const uint token =
                    (element / HeadDimension) % contextLength;
                const uint head =
                    element / (contextLength * HeadDimension);
                const uint slot = (contextStart + token) % Window;
                const ulong sourceKey =
                    ulong(head) * context_keys_strides[1]
                    + ulong(token) * context_keys_strides[2]
                    + ulong(dimension) * context_keys_strides[3];
                const ulong sourceValue =
                    ulong(head) * context_values_strides[1]
                    + ulong(token) * context_values_strides[2]
                    + ulong(dimension) * context_values_strides[3];
                const ulong destinationKey =
                    ulong(head) * ring_keys_strides[1]
                    + ulong(slot) * ring_keys_strides[2]
                    + ulong(dimension) * ring_keys_strides[3];
                const ulong destinationValue =
                    ulong(head) * ring_values_strides[1]
                    + ulong(slot) * ring_values_strides[2]
                    + ulong(dimension) * ring_values_strides[3];
                writableKeys[destinationKey] = context_keys[sourceKey];
                writableValues[destinationValue] =
                    context_values[sourceValue];
            }

            threadgroup float partialOutputs[Partitions * Partitions];
            threadgroup float maximumScores[Partitions];
            threadgroup float exponentialSums[Partitions];

            for (uint row = 0; row < Rows; ++row) {
                const ulong queryBase =
                    ulong(queryHead) * queries_strides[1]
                    + ulong(row) * queries_strides[2];
                float query[ValuesPerLane];
                float accumulated[ValuesPerLane];
                #pragma unroll
                for (uint index = 0; index < ValuesPerLane; ++index) {
                    const uint dimension = lane * ValuesPerLane + index;
                    query[index] = Scale * float(queries[
                        queryBase
                        + ulong(dimension) * queries_strides[3]]);
                    accumulated[index] = 0.0f;
                }

                float maximumScore = -3.402823466e+38f;
                float exponentialSum = 0.0f;
                for (uint logicalKey = partition;
                     logicalKey < totalLength;
                     logicalKey += Partitions) {
                    const bool proposal = logicalKey >= cachedLength;
                    const bool context =
                        logicalKey >= retainedLength && !proposal;
                    uint key = (context
                        ? logicalKey - retainedLength
                        : logicalKey);
                    if (proposal) key = logicalKey - cachedLength;
                    if (proposal
                        || cachedLength + row - logicalKey < Window) {
                        device const bfloat* keyData = ring_keys;
                        device const bfloat* valueData = ring_values;
                        ulong keyHeadStride = ring_keys_strides[1];
                        ulong keyRowStride = ring_keys_strides[2];
                        ulong keyDimensionStride = ring_keys_strides[3];
                        ulong valueHeadStride = ring_values_strides[1];
                        ulong valueRowStride = ring_values_strides[2];
                        ulong valueDimensionStride = ring_values_strides[3];
                        if (context) {
                            keyData = context_keys;
                            valueData = context_values;
                            keyHeadStride = context_keys_strides[1];
                            keyRowStride = context_keys_strides[2];
                            keyDimensionStride = context_keys_strides[3];
                            valueHeadStride = context_values_strides[1];
                            valueRowStride = context_values_strides[2];
                            valueDimensionStride = context_values_strides[3];
                        } else if (proposal) {
                            keyData = proposal_keys;
                            valueData = proposal_values;
                            keyHeadStride = proposal_keys_strides[1];
                            keyRowStride = proposal_keys_strides[2];
                            keyDimensionStride = proposal_keys_strides[3];
                            valueHeadStride = proposal_values_strides[1];
                            valueRowStride = proposal_values_strides[2];
                            valueDimensionStride =
                                proposal_values_strides[3];
                        } else {
                            key = (contextStart - retainedLength + key)
                                % Window;
                        }

                        const ulong keyBase =
                            ulong(kvHead) * keyHeadStride
                            + ulong(key) * keyRowStride;
                        const ulong valueBase =
                            ulong(kvHead) * valueHeadStride
                            + ulong(key) * valueRowStride;
                        float score = 0.0f;
                        #pragma unroll
                        for (uint index = 0;
                             index < ValuesPerLane; ++index) {
                            const uint dimension =
                                lane * ValuesPerLane + index;
                            score += query[index] * float(keyData[
                                keyBase
                                + ulong(dimension) * keyDimensionStride]);
                        }
                        score = simd_sum(score);

                        const float nextMaximum = max(maximumScore, score);
                        const float previousScale =
                            fast::exp(maximumScore - nextMaximum);
                        const float probability =
                            fast::exp(score - nextMaximum);
                        maximumScore = nextMaximum;
                        exponentialSum =
                            exponentialSum * previousScale + probability;
                        #pragma unroll
                        for (uint index = 0;
                             index < ValuesPerLane; ++index) {
                            const uint dimension =
                                lane * ValuesPerLane + index;
                            accumulated[index] =
                                accumulated[index] * previousScale
                                + probability * float(valueData[
                                    valueBase
                                    + ulong(dimension)
                                        * valueDimensionStride]);
                        }
                    }
                }

                if (lane == 0) {
                    maximumScores[partition] = maximumScore;
                    exponentialSums[partition] = exponentialSum;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                maximumScore = maximumScores[lane];
                const float globalMaximum = simd_max(maximumScore);
                const float partitionScale =
                    fast::exp(maximumScore - globalMaximum);
                exponentialSum = simd_sum(
                    exponentialSums[lane] * partitionScale);

                #pragma unroll
                for (uint index = 0; index < ValuesPerLane; ++index) {
                    partialOutputs[lane * Partitions + partition] =
                        accumulated[index];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    accumulated[index] = simd_sum(
                        partialOutputs[partition * Partitions + lane]
                            * partitionScale);
                    if (exponentialSum != 0.0f) {
                        accumulated[index] /= exponentialSum;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (lane == 0) {
                    const ulong outputBase =
                        (ulong(queryHead) * Rows + row) * HeadDimension;
                    #pragma unroll
                    for (uint index = 0;
                         index < ValuesPerLane; ++index) {
                        out[outputBase
                            + partition * ValuesPerLane + index] =
                            bfloat(accumulated[index]);
                    }
                }
            }
            """#,
        ensureRowContiguous: false
    )

    static func call(
        queries: MLXArray,
        ringKeys: MLXArray,
        ringValues: MLXArray,
        contextKeys: MLXArray,
        contextValues: MLXArray,
        proposalKeys: MLXArray,
        proposalValues: MLXArray,
        cacheState: MLXArray
    ) -> MLXArray {
        kernel(
            [
                queries,
                ringKeys, ringValues,
                contextKeys, contextValues,
                proposalKeys, proposalValues,
                cacheState,
            ],
            grid: (
                threadsPerThreadgroup * attentionHeads,
                1,
                1
            ),
            threadGroup: (threadsPerThreadgroup, 1, 1),
            outputShapes: [[1, attentionHeads, rows, headDimension]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}

enum Qwen38DFlash2RTN4M8Kernel {
    static let bits = 4
    static let groupSize = 64
    static let rows = 8
    static let tileN = 128
    private static let threadsPerThreadgroup = 256

    private static let kernel = MLXFast.metalKernel(
        name: "qwen38_dflash2_rtn4_mpp_m8_n128",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["out"],
        source: body,
        header: #"""
            #include <metal_stdlib>
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
            using namespace metal;
            using namespace mpp::tensor_ops;
            """#,
        ensureRowContiguous: false
    )

    static func call(
        _ x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        outputSize: Int
    ) -> MLXArray {
        let k = x.dim(-1)
        return kernel(
            [x, weight, scales, biases],
            template: [
                ("K", k),
                ("N", outputSize),
            ],
            grid: (
                threadsPerThreadgroup * outputSize / tileN,
                1,
                1
            ),
            threadGroup: (threadsPerThreadgroup, 1, 1),
            outputShapes: [[1, rows, outputSize]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    static func callPadded(
        _ x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        outputSize: Int
    ) -> MLXArray {
        let originalRows = x.dim(1)
        let input = originalRows == rows
            ? x.asType(.bfloat16)
            : concatenated(
                [
                    x.asType(.bfloat16),
                    MLXArray.zeros(
                        [1, rows - originalRows, x.dim(2)],
                        dtype: .bfloat16
                    ),
                ],
                axis: 1
            )
        let output = call(
            input,
            weight: weight,
            scales: scales,
            biases: biases,
            outputSize: outputSize
        )
        return originalRows == rows
            ? output
            : output[0..., ..<originalRows, 0...]
    }

    private static let body = #"""
        constexpr int TN = 128;
        constexpr int G = K / 64;

        const uint tile = threadgroup_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const uint simdGroup = simdgroup_index_in_threadgroup;
        const uint n0 = tile * TN;

        device bfloat* xp = const_cast<device bfloat*>(x);
        device uchar* wp = reinterpret_cast<device uchar*>(
            const_cast<device uint*>(w));

        auto a = tensor(
            xp,
            dextents<int, 2>{K, 8},
            array<int, 2>{1, K});
        auto c = tensor(
            out,
            dextents<int, 2>{N, 8},
            array<int, 2>{1, N});

        constexpr auto descriptor =
            matmul2d_descriptor(8, TN, 64, false, true, false);
        matmul2d<descriptor, execution_simdgroups<8>> operation;

        threadgroup float inputSums[8 * G];
        if (simdGroup < 8) {
            const uint row = simdGroup;
            for (uint group = 0; group < G; ++group) {
                const ulong offset =
                    ulong(row) * K + ulong(group) * 64 + lane;
                const float sum = simd_sum(
                    float(xp[offset]) + float(xp[offset + 32]));
                if (lane == 0) {
                    inputSums[group * 8 + row] = sum;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device uchar* firstWeight = wp + ulong(tile) * G * TN * 32;
        tensor<device uint4b_format, dextents<int, 2>, tensor_inline> firstB(
            firstWeight,
            dextents<int, 2>{64, TN},
            array<int, 2>{1, 64});
        auto a0 = a.slice<64, 8>(0, 0);
        auto b0 = firstB.slice<64, TN>(0, 0);
        auto accumulated = operation.template get_destination_cooperative_tensor<
            decltype(a0), decltype(b0), float>();

        #pragma unroll
        for (ushort index = 0; index < accumulated.get_capacity(); ++index) {
            accumulated[index] = 0.0f;
        }

        for (uint group = 0; group < G; ++group) {
            device uchar* groupWeight =
                wp + (ulong(tile) * G + group) * TN * 32;
            tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b(
                groupWeight,
                dextents<int, 2>{64, TN},
                array<int, 2>{1, 64});
            auto aSlice = a.slice<64, 8>(group * 64, 0);
            auto bSlice = b.slice<64, TN>(0, 0);
            auto partial = operation.template get_destination_cooperative_tensor<
                decltype(aSlice), decltype(bSlice), float>();
            operation.run(aSlice, bSlice, partial);

            #pragma unroll
            for (ushort index = 0; index < accumulated.get_capacity(); ++index) {
                auto coordinate = accumulated.get_multidimensional_index(index);
                const uint column = coordinate[0];
                const uint row = coordinate[1];
                const ulong parameter =
                    (ulong(tile) * G + group) * TN + column;
                accumulated[index] +=
                    partial[index] * float(scales[parameter])
                    + inputSums[group * 8 + row] * float(biases[parameter]);
            }
        }

        auto converted = operation.template get_destination_cooperative_tensor<
            decltype(a0), decltype(b0), bfloat>();
        #pragma unroll
        for (ushort index = 0; index < accumulated.get_capacity(); ++index) {
            converted[index] = bfloat(accumulated[index]);
        }
        converted.store(c.slice<TN, 8>(n0, 0));
        """#
}

enum Qwen38DFlash2RTN4M8GateUpKernel {
    private static let outputSize = 17_408
    private static let tileN = 256
    private static let persistentGroups = 36
    private static let threadsPerThreadgroup = 256

    private static let kernel = MLXFast.metalKernel(
        name: "qwen38_dflash2_rtn4_mpp_gate_up_m8_n256",
        inputNames: [
            "x",
            "gate_w", "gate_scales", "gate_biases",
            "up_w", "up_scales", "up_biases",
        ],
        outputNames: ["out"],
        source: body,
        header: #"""
            #include <metal_stdlib>
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
            using namespace metal;
            using namespace mpp::tensor_ops;
            """#,
        ensureRowContiguous: false
    )

    static func call(
        _ x: MLXArray,
        gate: Qwen38DFlash2Linear,
        up: Qwen38DFlash2Linear
    ) -> MLXArray {
        let gatePacked = gate.packed(tileN: tileN)
        let upPacked = up.packed(tileN: tileN)
        return kernel(
            [
                x.asType(.bfloat16),
                gatePacked.weight, gatePacked.scales, gatePacked.biases,
                upPacked.weight, upPacked.scales, upPacked.biases,
            ],
            template: [
                ("K", x.dim(-1)),
                ("N", outputSize),
                ("PG", persistentGroups),
            ],
            grid: (threadsPerThreadgroup * persistentGroups, 1, 1),
            threadGroup: (threadsPerThreadgroup, 1, 1),
            outputShapes: [[1, Qwen38DFlash2RTN4M8Kernel.rows, outputSize]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    private static let body = #"""
        constexpr uint TN = 256;
        constexpr uint G = K / 64;

        const uint lane = thread_index_in_simdgroup;
        const uint simdGroup = simdgroup_index_in_threadgroup;
        device bfloat* xp = const_cast<device bfloat*>(x);
        device uchar* gateWp = reinterpret_cast<device uchar*>(
            const_cast<device uint*>(gate_w));
        device uchar* upWp = reinterpret_cast<device uchar*>(
            const_cast<device uint*>(up_w));

        auto a = tensor(
            xp,
            dextents<int, 2>{K, 8},
            array<int, 2>{1, K});
        auto c = tensor(
            out,
            dextents<int, 2>{N, 8},
            array<int, 2>{1, N});
        constexpr auto descriptor =
            matmul2d_descriptor(8, TN, 64, false, true, false);
        matmul2d<descriptor, execution_simdgroups<8>> operation;

        threadgroup float inputSums[8 * G];
        if (simdGroup < 8) {
            const uint row = simdGroup;
            for (uint group = 0; group < G; ++group) {
                const ulong offset =
                    ulong(row) * K + ulong(group) * 64 + lane;
                const float sum = simd_sum(
                    float(xp[offset]) + float(xp[offset + 32]));
                if (lane == 0) {
                    inputSums[group * 8 + row] = sum;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tile = threadgroup_position_in_grid.x;
             tile < N / TN;
             tile += PG) {
            const uint outputOrigin = tile * TN;
            device uchar* firstGateWeight =
                gateWp + ulong(tile) * G * TN * 32;
            device uchar* firstUpWeight =
                upWp + ulong(tile) * G * TN * 32;
            tensor<device uint4b_format, dextents<int, 2>, tensor_inline>
                firstGate(
                    firstGateWeight,
                    dextents<int, 2>{64, TN},
                    array<int, 2>{1, 64});
            tensor<device uint4b_format, dextents<int, 2>, tensor_inline>
                firstUp(
                    firstUpWeight,
                    dextents<int, 2>{64, TN},
                    array<int, 2>{1, 64});
            auto a0 = a.slice<64, 8>(0, 0);
            auto gateB0 = firstGate.slice<64, TN>(0, 0);
            auto upB0 = firstUp.slice<64, TN>(0, 0);
            auto gateAccumulated = operation.template
                get_destination_cooperative_tensor<
                    decltype(a0), decltype(gateB0), float>();
            auto upAccumulated = operation.template
                get_destination_cooperative_tensor<
                    decltype(a0), decltype(upB0), float>();

            #pragma unroll
            for (ushort index = 0;
                 index < gateAccumulated.get_capacity(); ++index) {
                gateAccumulated[index] = 0.0f;
                upAccumulated[index] = 0.0f;
            }

            for (uint group = 0; group < G; ++group) {
                device uchar* gateGroupWeight = gateWp
                    + (ulong(tile) * G + group) * TN * 32;
                device uchar* upGroupWeight = upWp
                    + (ulong(tile) * G + group) * TN * 32;
                tensor<device uint4b_format, dextents<int, 2>, tensor_inline>
                    gateB(
                        gateGroupWeight,
                        dextents<int, 2>{64, TN},
                        array<int, 2>{1, 64});
                tensor<device uint4b_format, dextents<int, 2>, tensor_inline>
                    upB(
                        upGroupWeight,
                        dextents<int, 2>{64, TN},
                        array<int, 2>{1, 64});
                auto aSlice = a.slice<64, 8>(group * 64, 0);
                auto gateBSlice = gateB.slice<64, TN>(0, 0);
                auto upBSlice = upB.slice<64, TN>(0, 0);
                auto gatePartial = operation.template
                    get_destination_cooperative_tensor<
                        decltype(aSlice), decltype(gateBSlice), float>();
                auto upPartial = operation.template
                    get_destination_cooperative_tensor<
                        decltype(aSlice), decltype(upBSlice), float>();
                operation.run(aSlice, gateBSlice, gatePartial);
                operation.run(aSlice, upBSlice, upPartial);

                #pragma unroll
                for (ushort index = 0;
                     index < gateAccumulated.get_capacity(); ++index) {
                    auto coordinate =
                        gateAccumulated.get_multidimensional_index(index);
                    const uint column = coordinate[0];
                    const uint row = coordinate[1];
                    const ulong parameter =
                        (ulong(tile) * G + group) * TN + column;
                    const float inputSum = inputSums[group * 8 + row];
                    gateAccumulated[index] +=
                        gatePartial[index] * float(gate_scales[parameter])
                        + inputSum * float(gate_biases[parameter]);
                    upAccumulated[index] +=
                        upPartial[index] * float(up_scales[parameter])
                        + inputSum * float(up_biases[parameter]);
                }
            }

            auto converted = operation.template
                get_destination_cooperative_tensor<
                    decltype(a0), decltype(gateB0), bfloat>();
            #pragma unroll
            for (ushort index = 0;
                 index < gateAccumulated.get_capacity(); ++index) {
                const float gate = float(bfloat(gateAccumulated[index]));
                const float up = float(bfloat(upAccumulated[index]));
                converted[index] = bfloat(
                    gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * up);
            }
            converted.store(c.slice<TN, 8>(outputOrigin, 0));
        }
        """#
}

enum Qwen38DFlash2SelectorKernel {
    private static let positions = 7
    private static let topK = 16
    private static let shards = 8
    private static let threadsPerThreadgroup = 256

    private static let top16Header = #"""
        inline void qwen38_dflash2_top16_insert(
            thread float *values,
            thread uint *ids,
            float value,
            uint token)
        {
            if (!(value > values[15] ||
                  (value == values[15] && token < ids[15]))) return;
            uint slot = 15;
            while (slot > 0 &&
                   (value > values[slot - 1] ||
                    (value == values[slot - 1] && token < ids[slot - 1]))) {
                values[slot] = values[slot - 1];
                ids[slot] = ids[slot - 1];
                --slot;
            }
            values[slot] = value;
            ids[slot] = token;
        }
        """#

    private static let top16ShardedKernel = MLXFast.metalKernel(
        name: "qwen38_dflash2_selector_top16_sharded",
        inputNames: ["logits"],
        outputNames: ["partial_ids", "partial_values"],
        source: #"""
            constexpr uint Shards = 8;
            uint group = threadgroup_position_in_grid.x;
            uint position = group / Shards;
            uint shard = group % Shards;
            uint threadIndex = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simdGroup = simdgroup_index_in_threadgroup;
            uint vocabulary = uint(logits_shape[2]);
            threadgroup float groupValues[8 * 16];
            threadgroup uint groupIds[8 * 16];
            float localValues[16];
            uint localIds[16];
            for (uint index = 0; index < 16; ++index) {
                localValues[index] = -INFINITY;
                localIds[index] = 0xffffffffu;
            }
            for (uint token = shard * 256 + threadIndex;
                 token < vocabulary;
                 token += Shards * 256) {
                ulong offset = ulong(position) * ulong(logits_strides[1])
                    + ulong(token) * ulong(logits_strides[2]);
                qwen38_dflash2_top16_insert(
                    localValues, localIds, float(logits[offset]), token);
            }

            uint cursor = 0;
            for (uint rank = 0; rank < 16; ++rank) {
                float value = cursor < 16 ? localValues[cursor] : -INFINITY;
                uint token = cursor < 16 ? localIds[cursor] : 0xffffffffu;
                float simdBest = simd_max(value);
                uint simdId = simd_min(
                    value == simdBest ? token : 0xffffffffu);
                uint winner = simd_min(
                    value == simdBest && token == simdId
                        ? lane : 0xffffffffu);
                if (lane == 0) {
                    groupValues[simdGroup * 16 + rank] = simdBest;
                    groupIds[simdGroup * 16 + rank] = simdId;
                }
                if (lane == winner) ++cursor;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (threadIndex == 0) {
                float finalValues[16];
                uint finalIds[16];
                for (uint index = 0; index < 16; ++index) {
                    finalValues[index] = -INFINITY;
                    finalIds[index] = 0xffffffffu;
                }
                for (uint item = 0; item < 8 * 16; ++item) {
                    qwen38_dflash2_top16_insert(
                        finalValues, finalIds,
                        groupValues[item], groupIds[item]);
                }
                for (uint rank = 0; rank < 16; ++rank) {
                    partial_ids[group * 16 + rank] = int(finalIds[rank]);
                    partial_values[group * 16 + rank] = finalValues[rank];
                }
            }
            """#,
        header: top16Header,
        ensureRowContiguous: false
    )

    private static let top16ReduceKernel = MLXFast.metalKernel(
        name: "qwen38_dflash2_selector_top16_reduce",
        inputNames: ["partial_ids", "partial_values"],
        outputNames: ["candidates", "unary"],
        source: #"""
            if (thread_position_in_threadgroup.x != 0) return;
            constexpr uint Shards = 8;
            uint position = threadgroup_position_in_grid.x;
            float values[16];
            uint ids[16];
            for (uint index = 0; index < 16; ++index) {
                values[index] = -INFINITY;
                ids[index] = 0xffffffffu;
            }
            uint origin = position * Shards * 16;
            for (uint item = 0; item < Shards * 16; ++item) {
                qwen38_dflash2_top16_insert(
                    values, ids,
                    partial_values[origin + item],
                    uint(partial_ids[origin + item]));
            }
            for (uint rank = 0; rank < 16; ++rank) {
                candidates[position * 16 + rank] = int(ids[rank]);
                unary[position * 16 + rank] = bfloat(values[rank]);
            }
            """#,
        header: top16Header,
        ensureRowContiguous: false
    )

    private static let greedyKernel = MLXFast.metalKernel(
        name: "qwen38_dflash2_selector_greedy",
        inputNames: [
            "hidden", "candidates", "unary", "predecessor_codebook",
            "successor_codebook", "anchor",
        ],
        outputNames: ["tokens"],
        source: #"""
            uint threadIndex = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simdGroup = simdgroup_index_in_threadgroup;
            uint vocabulary = uint(predecessor_codebook_shape[0]);
            uint rank = uint(predecessor_codebook_shape[1]);
            uint fallback = vocabulary > 248070u ? 248070u : 0u;
            threadgroup float scores[16];
            threadgroup uint predecessor;
            if (threadIndex == 0) predecessor = uint(anchor[0]);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint position = 0; position < 7; ++position) {
                for (uint wave = 0; wave < 2; ++wave) {
                    uint candidateIndex = wave * 8 + simdGroup;
                    uint candidate = uint(
                        candidates[position * 16 + candidateIndex]);
                    uint safeCandidate =
                        candidate < vocabulary ? candidate : fallback;
                    uint safePredecessor =
                        predecessor < vocabulary ? predecessor : fallback;
                    float edge = 0.0f;
                    for (uint dimension = lane; dimension < rank;
                         dimension += 32) {
                        ulong predecessorOffset =
                            ulong(safePredecessor)
                                * ulong(predecessor_codebook_strides[0])
                            + ulong(dimension)
                                * ulong(predecessor_codebook_strides[1]);
                        ulong hiddenOffset =
                            ulong(position) * ulong(hidden_strides[1])
                            + ulong(dimension) * ulong(hidden_strides[2]);
                        ulong successorOffset =
                            ulong(safeCandidate)
                                * ulong(successor_codebook_strides[0])
                            + ulong(dimension)
                                * ulong(successor_codebook_strides[1]);
                        edge += float(predecessor_codebook[predecessorOffset])
                            * float(hidden[hiddenOffset])
                            * float(successor_codebook[successorOffset]);
                    }
                    edge = simd_sum(edge);
                    if (lane == 0) {
                        scores[candidateIndex] =
                            float(unary[position * 16 + candidateIndex]) + edge;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (threadIndex == 0) {
                    uint best = 0;
                    for (uint index = 1; index < 16; ++index) {
                        if (scores[index] > scores[best]) best = index;
                    }
                    predecessor = uint(candidates[position * 16 + best]);
                    tokens[position] = int(predecessor);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """#,
        ensureRowContiguous: false
    )

    static func call(
        hidden: MLXArray,
        logits: MLXArray,
        predecessorCodebook: MLXArray,
        successorCodebook: MLXArray,
        anchor: MLXArray
    ) -> MLXArray {
        let partials = top16ShardedKernel(
            [logits],
            grid: (positions * shards * threadsPerThreadgroup, 1, 1),
            threadGroup: (threadsPerThreadgroup, 1, 1),
            outputShapes: [
                [positions, shards, topK],
                [positions, shards, topK],
            ],
            outputDTypes: [.int32, .float32]
        )
        let reduced = top16ReduceKernel(
            partials,
            grid: (positions * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[positions, topK], [positions, topK]],
            outputDTypes: [.int32, .bfloat16]
        )
        return greedyKernel(
            [
                hidden, reduced[0], reduced[1], predecessorCodebook,
                successorCodebook, anchor,
            ],
            grid: (threadsPerThreadgroup, 1, 1),
            threadGroup: (threadsPerThreadgroup, 1, 1),
            outputShapes: [[1, positions]],
            outputDTypes: [.int32]
        )[0]
    }
}

final class Qwen38DFlash2PackedRTN4 {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray

    init(weight: MLXArray, scales: MLXArray, biases: MLXArray) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
    }
}

final class Qwen38DFlash2Linear: QuantizedLinear {
    private let packed128: Qwen38DFlash2PackedRTN4
    private var packed256: Qwen38DFlash2PackedRTN4?
    private let quantizationBiases: MLXArray
    private let outputSize: Int

    var packedArrays: [MLXArray] {
        [packed128.weight, packed128.scales, packed128.biases]
    }

    init(
        weight loadedWeight: MLXArray,
        scales loadedScales: MLXArray,
        biases loadedBiases: MLXArray
    ) {
        let n = loadedWeight.dim(0)
        packed128 = Self.makePacked(
            weight: loadedWeight,
            scales: loadedScales,
            biases: loadedBiases,
            tileN: Qwen38DFlash2RTN4M8Kernel.tileN
        )
        quantizationBiases = loadedBiases
        outputSize = n

        super.init(
            weight: loadedWeight,
            scales: loadedScales,
            biases: loadedBiases,
            groupSize: Qwen38DFlash2RTN4M8Kernel.groupSize,
            bits: Qwen38DFlash2RTN4M8Kernel.bits,
            mode: .affine
        )
    }

    func packed(tileN: Int) -> Qwen38DFlash2PackedRTN4 {
        if tileN == Qwen38DFlash2RTN4M8Kernel.tileN {
            return packed128
        }
        if let packed256 {
            return packed256
        }
        let packed = Self.makePacked(
            weight: weight,
            scales: scales,
            biases: quantizationBiases,
            tileN: tileN
        )
        packed256 = packed
        return packed
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let originalRows = x.ndim == 3 ? x.dim(1) : 0
        if x.ndim == 3,
           x.dim(0) == 1,
           originalRows > 0,
           originalRows <= Qwen38DFlash2RTN4M8Kernel.rows,
           x.dim(2) % Qwen38DFlash2RTN4M8Kernel.groupSize == 0
        {
            let input = originalRows == Qwen38DFlash2RTN4M8Kernel.rows
                ? x.asType(.bfloat16)
                : concatenated(
                    [
                        x.asType(.bfloat16),
                        MLXArray.zeros(
                            [
                                1,
                                Qwen38DFlash2RTN4M8Kernel.rows - originalRows,
                                x.dim(2),
                            ],
                            dtype: .bfloat16
                        ),
                    ],
                    axis: 1
                )
            var output = Qwen38DFlash2RTN4M8Kernel.call(
                input,
                weight: packed128.weight,
                scales: packed128.scales,
                biases: packed128.biases,
                outputSize: outputSize
            )
            if originalRows != Qwen38DFlash2RTN4M8Kernel.rows {
                output = output[0..., ..<originalRows, 0...]
            }
            if let bias { output = output + bias }
            return output
        }

        var output = quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        if let bias { output = output + bias }
        return output
    }

    private static func makePacked(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        tileN: Int
    ) -> Qwen38DFlash2PackedRTN4 {
        let n = weight.dim(0)
        let groups = scales.dim(1)
        precondition(n % tileN == 0)

        let tiledWeight = weight
            .view(dtype: .uint8)
            .reshaped(n / tileN, tileN, groups, 32)
            .swappedAxes(1, 2)
            .contiguous()
            .view(dtype: .uint32)
        let tiledScales = scales
            .asType(.bfloat16)
            .reshaped(n / tileN, tileN, groups)
            .swappedAxes(1, 2)
            .contiguous()
        let tiledBiases = biases
            .asType(.bfloat16)
            .reshaped(n / tileN, tileN, groups)
            .swappedAxes(1, 2)
            .contiguous()
        return Qwen38DFlash2PackedRTN4(
            weight: tiledWeight,
            scales: tiledScales,
            biases: tiledBiases
        )
    }
}
