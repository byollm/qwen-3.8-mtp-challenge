import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct Qwen38DraftConfiguration: Sendable {
    let targetLayerIDs = [5, 19, 33, 47, 61]
    let hiddenLayers = 5
    let hiddenSize = 5_120
    let intermediateSize = 17_408
    let vocabularySize = 248_320
    let attentionHeads = 32
    let kvHeads = 8
    let headDim = 128
    let slidingWindow = 2_048
    let convolutionKernelSize = 2
    let convolutionGroupSize = 16
    let selectorRank = 256
    let selectorTopK = 16
    let maskTokenID: Int32 = 248_070

    let rmsNormEpsilon: Float = 1e-6
    let maxPositionEmbeddings = 262_144
    let ropeTheta: Float = 10_000_000
}

struct Qwen38DraftContextPlan {
    let hidden: MLXArray
    let startOffset: Int
    let endOffset: Int
}

struct Qwen38DraftCache {
    private(set) var offset = 0
    private var length = 0
    private var keys: MLXArray?
    private var values: MLXArray?

    mutating func planAppend(
        context: MLXArray,
        window: Int
    ) -> Qwen38DraftContextPlan {
        let retainedLength = window - 1
        let skippedLength = keys == nil
            ? max(0, context.dim(1) - retainedLength)
            : 0
        let hidden = skippedLength == 0
            ? context
            : context[0..., skippedLength..., 0...]

        offset += skippedLength
        return Qwen38DraftContextPlan(
            hidden: hidden,
            startOffset: offset,
            endOffset: offset + hidden.dim(1)
        )
    }

    mutating func append(
        keys: MLXArray,
        values: MLXArray,
        plan: Qwen38DraftContextPlan,
        window: Int
    ) -> (keys: MLXArray, values: MLXArray, state: MLXArray) {
        let retainedLength = min(length, window - 2)
        if self.keys == nil {
            let shape = [1, keys.dim(1), window, keys.dim(3)]
            self.keys = MLXArray.zeros(shape, dtype: keys.dtype)
            self.values = MLXArray.zeros(shape, dtype: values.dtype)
        }

        length = retainedLength + keys.dim(2)
        offset = plan.endOffset
        return (
            self.keys!,
            self.values!,
            MLXArray([Int32(retainedLength), Int32(plan.startOffset)])
        )
    }
}

final class Qwen38DraftAttention: Module {
    private let attentionHeads: Int
    private let kvHeads: Int
    private let slidingWindow: Int
    private var packedQKV: Qwen38DFlash2PackedRTN4?
    private var affineQKVWeight: MLXArray?
    private var affineQKVScales: MLXArray?
    private var affineQKVBiases: MLXArray?

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    init(_ configuration: Qwen38DraftConfiguration) {
        self.attentionHeads = configuration.attentionHeads
        self.kvHeads = configuration.kvHeads
        self.slidingWindow = configuration.slidingWindow
        _queryProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.attentionHeads * configuration.headDim,
            bias: false
        )
        _keyProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.kvHeads * configuration.headDim,
            bias: false
        )
        _valueProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.kvHeads * configuration.headDim,
            bias: false
        )
        _outputProjection.wrappedValue = Linear(
            configuration.attentionHeads * configuration.headDim,
            configuration.hiddenSize,
            bias: false
        )
        _queryNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDim,
            eps: configuration.rmsNormEpsilon
        )
        _keyNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDim,
            eps: configuration.rmsNormEpsilon
        )
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        context: MLXArray,
        rope: RoPELayer,
        cache: inout Qwen38DraftCache
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let proposalLength = hiddenStates.dim(1)
        let plan = cache.planAppend(context: context, window: slidingWindow)
        let contextLength = plan.hidden.dim(1)
        let queryWidth = attentionHeads * 128
        let keyWidth = kvHeads * 128
        let proposalQKV = qkv(hiddenStates)
        let contextQKV = qkv(plan.hidden)

        var queries = queryNorm(
            proposalQKV[0..., 0..., ..<queryWidth].reshaped(
                batchSize,
                proposalLength,
                attentionHeads,
                -1
            )
        ).transposed(0, 2, 1, 3)
        var contextKeys = keyNorm(
            contextQKV[0..., 0..., queryWidth..<(queryWidth + keyWidth)].reshaped(
                batchSize,
                contextLength,
                kvHeads,
                -1
            )
        ).transposed(0, 2, 1, 3)
        let contextValues = contextQKV[0..., 0..., (queryWidth + keyWidth)...]
            .reshaped(batchSize, contextLength, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        var proposalKeys = keyNorm(
            proposalQKV[0..., 0..., queryWidth..<(queryWidth + keyWidth)].reshaped(
                batchSize,
                proposalLength,
                kvHeads,
                -1
            )
        ).transposed(0, 2, 1, 3)
        let proposalValues = proposalQKV[0..., 0..., (queryWidth + keyWidth)...]
            .reshaped(batchSize, proposalLength, kvHeads, -1)
            .transposed(0, 2, 1, 3)

        queries = rope(queries, offset: plan.endOffset)
        contextKeys = rope(contextKeys, offset: plan.startOffset)
        proposalKeys = rope(proposalKeys, offset: plan.endOffset)

        let cached = cache.append(
            keys: contextKeys,
            values: contextValues,
            plan: plan,
            window: slidingWindow
        )

        return outputProjection(
            Qwen38DFlash2AttentionKernel.call(
                queries: queries,
                ringKeys: cached.keys,
                ringValues: cached.values,
                contextKeys: contextKeys,
                contextValues: contextValues,
                proposalKeys: proposalKeys,
                proposalValues: proposalValues,
                cacheState: cached.state
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, proposalLength, -1)
        )
    }

    private func qkv(_ hiddenStates: MLXArray) -> MLXArray {
        if hiddenStates.dim(1) > Qwen38DFlash2RTN4M8Kernel.rows {
            let weight: MLXArray
            let scales: MLXArray
            let biases: MLXArray
            if let affineQKVWeight, let affineQKVScales, let affineQKVBiases {
                weight = affineQKVWeight
                scales = affineQKVScales
                biases = affineQKVBiases
            } else {
                let query = queryProjection as! Qwen38DFlash2Linear
                let key = keyProjection as! Qwen38DFlash2Linear
                let value = valueProjection as! Qwen38DFlash2Linear
                weight = concatenated(
                    [query.weight, key.weight, value.weight], axis: 0
                ).contiguous()
                scales = concatenated(
                    [query.scales, key.scales, value.scales], axis: 0
                ).contiguous()
                biases = concatenated(
                    [query.biases!, key.biases!, value.biases!], axis: 0
                ).contiguous()
                affineQKVWeight = weight
                affineQKVScales = scales
                affineQKVBiases = biases
            }
            return quantizedMM(
                hiddenStates,
                weight,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: Qwen38DFlash2RTN4M8Kernel.groupSize,
                bits: Qwen38DFlash2RTN4M8Kernel.bits,
                mode: .affine
            )
        }

        let packed: Qwen38DFlash2PackedRTN4
        if let packedQKV {
            packed = packedQKV
        } else {
            let queryPacked =
                (queryProjection as! Qwen38DFlash2Linear).packed(
                    tileN: Qwen38DFlash2RTN4M8Kernel.tileN)
            let keyPacked =
                (keyProjection as! Qwen38DFlash2Linear).packed(
                    tileN: Qwen38DFlash2RTN4M8Kernel.tileN)
            let valuePacked =
                (valueProjection as! Qwen38DFlash2Linear).packed(
                    tileN: Qwen38DFlash2RTN4M8Kernel.tileN)
            packed = Qwen38DFlash2PackedRTN4(
                weight: concatenated(
                    [queryPacked.weight, keyPacked.weight, valuePacked.weight],
                    axis: 0
                ),
                scales: concatenated(
                    [queryPacked.scales, keyPacked.scales, valuePacked.scales],
                    axis: 0
                ),
                biases: concatenated(
                    [queryPacked.biases, keyPacked.biases, valuePacked.biases],
                    axis: 0
                )
            )
            packedQKV = packed
        }
        return Qwen38DFlash2RTN4M8Kernel.callPadded(
            hiddenStates,
            weight: packed.weight,
            scales: packed.scales,
            biases: packed.biases,
            outputSize: attentionHeads * 128 + 2 * kvHeads * 128
        )
    }
}

final class Qwen38DraftMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(_ configuration: Qwen38DraftConfiguration) {
        _gateProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.intermediateSize,
            bias: false
        )
        _upProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.intermediateSize,
            bias: false
        )
        _downProjection.wrappedValue = Linear(
            configuration.intermediateSize,
            configuration.hiddenSize,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(
            Qwen38DFlash2RTN4M8GateUpKernel.call(
                hiddenStates,
                gate: gateProjection as! Qwen38DFlash2Linear,
                up: upProjection as! Qwen38DFlash2Linear
            )
        )
    }
}

final class Qwen38GroupedDynamicCausalConv: Module {
    private let kernelSize: Int
    private let groups: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        self.kernelSize = kernelSize
        self.groups = hiddenSize / groupSize
        _baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(
            hiddenSize,
            2 * kernelSize * groups,
            bias: false
        )
        super.init()
    }

    func prepare(_ hiddenStates: MLXArray) -> (hidden: MLXArray, dynamic: MLXArray) {
        let dynamic = kernelProjection(hiddenStates).reshaped(
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            2,
            kernelSize,
            groups
        )
        return (
            Qwen38DFlash2ConvKernel.call(
                hiddenStates,
                dynamic: dynamic,
                base: baseKernel,
                residual: hiddenStates,
                fuseResidual: false
            ),
            dynamic
        )
    }

    func finish(
        _ hiddenStates: MLXArray,
        dynamic: MLXArray,
        residual: MLXArray
    ) -> MLXArray {
        Qwen38DFlash2ConvKernel.call(
            hiddenStates,
            dynamic: dynamic,
            base: baseKernel,
            residual: residual,
            fuseResidual: true
        )
    }
}

final class Qwen38DraftDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Qwen38DraftAttention
    @ModuleInfo(key: "attention_conv")
    var attentionConvolution: Qwen38GroupedDynamicCausalConv
    @ModuleInfo(key: "mlp") var mlp: Qwen38DraftMLP
    @ModuleInfo(key: "mlp_conv") var mlpConvolution: Qwen38GroupedDynamicCausalConv
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ configuration: Qwen38DraftConfiguration) {
        _selfAttention.wrappedValue = Qwen38DraftAttention(configuration)
        _attentionConvolution.wrappedValue = Qwen38GroupedDynamicCausalConv(
            hiddenSize: configuration.hiddenSize,
            kernelSize: configuration.convolutionKernelSize,
            groupSize: configuration.convolutionGroupSize
        )
        _mlp.wrappedValue = Qwen38DraftMLP(configuration)
        _mlpConvolution.wrappedValue = Qwen38GroupedDynamicCausalConv(
            hiddenSize: configuration.hiddenSize,
            kernelSize: configuration.convolutionKernelSize,
            groupSize: configuration.convolutionGroupSize
        )
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        context: MLXArray,
        rope: RoPELayer,
        cache: inout Qwen38DraftCache
    ) -> MLXArray {
        let attentionPrepared = attentionConvolution.prepare(
            inputLayerNorm(hiddenStates)
        )
        let afterAttention = attentionConvolution.finish(
            selfAttention(
                attentionPrepared.hidden,
                context: context,
                rope: rope,
                cache: &cache
            ),
            dynamic: attentionPrepared.dynamic,
            residual: hiddenStates
        )
        let mlpPrepared = mlpConvolution.prepare(
            postAttentionLayerNorm(afterAttention)
        )
        return mlpConvolution.finish(
            mlp(mlpPrepared.hidden),
            dynamic: mlpPrepared.dynamic,
            residual: afterAttention
        )
    }
}

final class Qwen38CandidateSelector: Module {
    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    init(_ configuration: Qwen38DraftConfiguration) {
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.selectorRank
        )
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.selectorRank
        )
        _hiddenProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.selectorRank,
            bias: false
        )
        super.init()
    }

    func select(
        hiddenStates: MLXArray,
        logits: MLXArray,
        anchor: MLXArray
    ) -> MLXArray {
        let projectedHidden = hiddenProjection(hiddenStates)
        return Qwen38DFlash2SelectorKernel.call(
            hidden: projectedHidden,
            logits: logits,
            predecessorCodebook: predecessorCodebook.weight,
            successorCodebook: successorCodebook.weight,
            anchor: anchor
        )
    }
}

struct Qwen38DraftState {
    var caches: [Qwen38DraftCache]
}

final class Qwen38DraftModel: Module {
    static let blockSize = 8
    static let proposalCount = 7

    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Qwen38DraftDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector: Qwen38CandidateSelector

    private let configuration: Qwen38DraftConfiguration
    private let rope: RoPELayer

    init(_ configuration: Qwen38DraftConfiguration = Qwen38DraftConfiguration()) {
        self.configuration = configuration
        _fc.wrappedValue = Linear(
            configuration.targetLayerIDs.count * configuration.hiddenSize,
            configuration.hiddenSize,
            bias: false
        )
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map { _ in
            Qwen38DraftDecoderLayer(configuration)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        _candidateSelector.wrappedValue = Qwen38CandidateSelector(configuration)
        self.rope = initializeRope(
            dims: configuration.headDim,
            base: configuration.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: configuration.maxPositionEmbeddings
        )
        super.init()
    }

    func makeState() -> Qwen38DraftState {
        Qwen38DraftState(
            caches: (0 ..< configuration.hiddenLayers).map { _ in
                Qwen38DraftCache()
            }
        )
    }

    func propose(
        anchor: MLXArray,
        targetTaps: MLXArray,
        embedding: (MLXArray) -> MLXArray,
        head: (MLXArray) -> MLXArray,
        state: inout Qwen38DraftState
    ) -> MLXArray {
        let inputIDs = concatenated(
            [
                anchor.reshaped(1, 1),
                MLXArray(
                    Array(
                        repeating: configuration.maskTokenID,
                        count: Self.proposalCount
                    )
                ).reshaped(1, Self.proposalCount),
            ],
            axis: 1
        )
        var hiddenStates = embedding(inputIDs).asType(.bfloat16)
        let context = hiddenNorm(fc(targetTaps.asType(.bfloat16)))
        for layerIndex in 0 ..< configuration.hiddenLayers {
            hiddenStates = layers[layerIndex](
                hiddenStates,
                context: context,
                rope: rope,
                cache: &state.caches[layerIndex]
            )
        }
        hiddenStates = norm(hiddenStates)

        let liveHidden = hiddenStates[0..., 1 ..< Self.blockSize, 0...]
        let headInput = concatenated(
            [
                liveHidden.asType(.bfloat16),
                MLXArray.zeros(
                    [1, 1, configuration.hiddenSize],
                    dtype: .bfloat16
                ),
            ],
            axis: 1
        )
        let logits = head(headInput)[0..., ..<Self.proposalCount, 0...]
        return candidateSelector.select(
            hiddenStates: liveHidden,
            logits: logits,
            anchor: anchor
        )
    }
}
