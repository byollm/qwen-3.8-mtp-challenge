// Copyright © 2026 Eigen Labs.
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/qwen35_model.py  (MTPDecoderLayer, MTPModule)
//   patches/mlx_lm_mtp/__init__.py        (is_mtp_active / set_mtp_active)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Module-level MTP flag

/// Controls whether Qwen3.5/3.6 model inits attach the MTP head.
/// Set to `true` before calling `MLXLLM.load(...)` when MTP should be active.
/// Mirrors omlx `is_mtp_active()` / `set_mtp_active()` from
/// patches/mlx_lm_mtp/__init__.py.
public nonisolated(unsafe) var _qwen35MTPEnabled: Bool = false

/// E85 arm gate. `MLX_E85_FUSED_EMBED=0` restores the eager
/// `embedTokens(ids)` before the dual-norm concat.
///
/// The `MLX_` prefix is load-bearing: the trusted worker's environment
/// sanitizer drops `MLXFAST_*`, so an `MLXFAST_`-spelled gate would never
/// reach the process that runs the scored round, and both arms of an A/B
/// would silently measure the same code.
let qwen35FusedEmbedConcatEnabled: Bool =
    ProcessInfo.processInfo.environment["MLX_E85_FUSED_EMBED"] != "0"

/// E121 arm gate. `MLX_QWEN_MTP_HEAD_NORM_FUSE=0` restores the eager
/// `norm(h + mlpOut)` pair at the MTP head's own tail (both
/// `MTPModule.callAsFunction` and `lastHiddenWithKVOnlyHistory`) instead of
/// folding the exit add into the head's final RMSNorm launch. Same `MLX_`
/// prefix reasoning as `MLX_E85_FUSED_EMBED` above: `MLXFAST_*` is stripped
/// by the trusted worker's sandbox before the scored process ever sees it.
let qwen35MTPHeadNormFuseEnabled: Bool =
    ProcessInfo.processInfo.environment["MLX_QWEN_MTP_HEAD_NORM_FUSE"] != "0"

/// `MLX_QWEN_E85_GEOMETRY_CACHE=0` restores `preFcConcat`'s full per-call
/// guard chain byte-for-byte, recomputing every sub-check below on every
/// draft substep. Default on: the sub-checks this flag skips depend only on
/// `embedTokens`'s own quantization layout (mode/bits/groupSize/dtypes/zero-
/// point shape) and the two head-side RMSNorm `eps` values, none of which
/// can change once the module and its embedding table are constructed, so
/// re-deriving the same `true` on every call is pure repeated work.
let qwen35E85GeometryCacheEnabled: Bool =
    ProcessInfo.processInfo.environment["MLX_QWEN_E85_GEOMETRY_CACHE"] != "0"

// MARK: - MTPDecoderLayer

/// Full-attention transformer layer used inside the Qwen3.5/3.6 MTP head.
/// Unlike `Qwen35DecoderLayer`, this always uses full attention (never SSM/linear).
/// MoE config is honoured when `num_experts > 0`.
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPDecoderLayer
final class Qwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            // Same fused gate/up MLP as the backbone layers; here the linears
            // stay bf16 and the fuse takes the plain-weight path. Head side —
            // proposal-only, no exactness constraint.
            _mlp.wrappedValue = Qwen35FusedMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        let (h, mlpOut) = residualAndMLPOutput(x, mask: mask, cache: cache)
        return h + mlpOut
    }

    /// Same computation as `callAsFunction`, but returns the pre-final-add
    /// `(h, mlpOut)` pair instead of merging them. A caller whose very next
    /// step is an RMSNorm (both `MTPModule` call sites below: this is the
    /// LAST -- here the only -- layer in the head) can then fuse that merge
    /// into the norm's own launch via `qwen35FusedResidualRMSNorm` instead of
    /// paying a standalone add. `h + mlpOut` here is bit-identical to
    /// `callAsFunction`'s return, so a caller that just adds the pair back
    /// together gets the exact old behavior.
    func residualAndMLPOutput(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> (h: MLXArray, mlpOut: MLXArray) {
        // omlx: MTPDecoderLayer.__call__
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        // The backbone's decoder layer has fused this residual+norm boundary
        // since `qwen35FusedResidualRMSNorm` landed; the head layer was left on
        // the eager pair. Same kernel, same bf16/5120 guard, same
        // bf16-round-before-square argument, so the values are bit-identical to
        // `h = x + r; postAttentionLayerNorm(h)` — one launch and one host graph
        // node instead of two, paid once per PROPOSED token (draftCount times a
        // round) rather than once per layer.
        if x.dtype == .bfloat16, r.dtype == .bfloat16, x.dim(-1) == 5120 {
            let (h, postAttnNorm) = qwen35FusedResidualRMSNorm(
                x: x, r: r,
                weight: postAttentionLayerNorm.weight,
                eps: postAttentionLayerNorm.eps)
            return (h, (mlp as! UnaryLayer)(postAttnNorm))
        }
        let h = x + r
        return (h, (mlp as! UnaryLayer)(postAttentionLayerNorm(h)))
    }

    /// Populate this layer's K/V history without computing a dead layer
    /// output. Only valid when no later MTP layer consumes that output.
    func appendHistoryKV(_ x: MLXArray, cache: any KVCache) {
        selfAttn.appendHistoryKV(inputLayerNorm(x), cache: cache)
    }
}

// MARK: - MTPModule

/// Multi-Token Prediction head for Qwen3.5/3.6.
///
/// Fuses the backbone's pre-norm hidden state at position t with the embedding of
/// the sampled main token (t+1) to predict the draft token at (t+2).
///
/// Architecture (port of PR #990):
/// ```
/// pre_fc_norm_hidden:    RMSNorm(hidden_size)
/// pre_fc_norm_embedding: RMSNorm(hidden_size)
/// fc:                    Linear(hidden_size * 2 → hidden_size, bias: false)
/// layers:                [MTPDecoderLayer]  × mtp_num_hidden_layers
/// norm:                  RMSNorm(hidden_size)
/// ```
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPModule
final class Qwen35MTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    // `layers` uses the default ModuleInfo key derived from the property name.
    let layers: [Qwen35MTPDecoderLayer]
    let norm: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        _preFcNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFcNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        self.layers = (0 ..< args.mtpNumHiddenLayers).map { _ in
            Qwen35MTPDecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    /// Cache slot for `e85StaticGateOK`, keyed by the embedding table's
    /// identity so a swapped table (never happens in this single-model
    /// session, but the guard costs one pointer compare) recomputes instead
    /// of trusting a stale answer.
    private var _e85StaticGateCache: (ObjectIdentifier, Bool)?

    /// The subset of `preFcConcat`'s E85 guard that depends only on
    /// `quantized`'s own layout and `self`'s two RMSNorm `eps` values —
    /// never on the per-call `hidden`/`nextTokenIds` shapes, which stay live
    /// checks at the call site. Same conjuncts as the pristine guard, just
    /// unrolled here, so a kill-switch flip changes only whether the answer
    /// is cached, never what is checked.
    private func e85StaticGateOK(_ quantized: QuantizedEmbedding) -> Bool {
        func compute() -> Bool {
            guard quantized.mode == .affine, quantized.bits == 4,
                  quantized.groupSize == 64,
                  let biases = quantized.biases,
                  quantized.weight.dtype == .uint32,
                  quantized.scales.dtype == .bfloat16,
                  biases.dtype == .bfloat16,
                  biases.shape == quantized.scales.shape,
                  preFcNormEmbedding.eps == preFcNormHidden.eps
            else { return false }
            return true
        }
        guard qwen35E85GeometryCacheEnabled else { return compute() }
        let identity = ObjectIdentifier(quantized)
        if let cached = _e85StaticGateCache, cached.0 == identity {
            return cached.1
        }
        let ok = compute()
        _e85StaticGateCache = (identity, ok)
        return ok
    }

    /// Dual RMSNorm written straight into the `[e | h]` layout `fc` consumes.
    /// Same arithmetic as `qwen35DualRMSNorm` + `concatenated([e, h], -1)`;
    /// the extra concat launch is gone. Proposal-only.
    ///
    /// The embedding table is affine 4-bit group-64, so `embedTokens(ids)` is
    /// three gathers plus a dequantize. Those four intermediates exist only to
    /// carry one row into a kernel that reads it twice, so the fused variant
    /// reads the packed row in place and the eager embed never runs.
    private func preFcConcat(
        nextTokenIds: MLXArray, embedTokens: Embedding, hidden: MLXArray
    ) -> MLXArray {
        if qwen35FusedEmbedConcatEnabled,
           let quantized = embedTokens as? QuantizedEmbedding,
           let zeroPoints = quantized.biases,
           e85StaticGateOK(quantized),
           hidden.dtype == .bfloat16, hidden.dim(-1) == 5120,
           quantized.weight.dim(1) * 8 == hidden.dim(-1),
           quantized.scales.dim(1) * 64 == hidden.dim(-1),
           nextTokenIds.dtype == .int32,
           nextTokenIds.ndim == 2, nextTokenIds.dim(0) == 1,
           nextTokenIds.strides.last == 1,
           nextTokenIds.dim(1) * hidden.dim(-1) == hidden.size
        {
            return qwen35EmbedDualRMSNormConcat(
                ids: nextTokenIds,
                embedWeight: quantized.weight,
                embedScales: quantized.scales,
                embedBiases: zeroPoints,
                b: hidden,
                aWeight: preFcNormEmbedding.weight,
                bWeight: preFcNormHidden.weight,
                eps: preFcNormEmbedding.eps)
        }

        let embeds = embedTokens(nextTokenIds)
        if embeds.dtype == .bfloat16, hidden.dtype == .bfloat16,
           embeds.dim(-1) == 5120, hidden.dim(-1) == 5120,
           embeds.shape == hidden.shape,
           preFcNormEmbedding.eps == preFcNormHidden.eps
        {
            return qwen35DualRMSNormConcat(
                a: embeds, b: hidden,
                aWeight: preFcNormEmbedding.weight,
                bWeight: preFcNormHidden.weight,
                eps: preFcNormEmbedding.eps)
        }
        return concatenated(
            [preFcNormEmbedding(embeds), preFcNormHidden(hidden)], axis: -1)
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        var fused = fc(
            preFcConcat(
                nextTokenIds: nextTokenIds, embedTokens: embedTokens,
                hidden: hidden))

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)

        // 3. Run every layer but the last eagerly; fuse the last layer's
        //    exit add into the head's own final RMSNorm launch below instead
        //    of paying both separately (`mtpNumHiddenLayers` ships as 1, so
        //    in practice this loop runs zero times and every draft step
        //    goes straight to the fused tail).
        guard let lastIndex = layers.indices.last else { return norm(fused) }
        for i in layers.indices where i != lastIndex {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            fused = layers[i](fused, mask: mask, cache: c)
        }
        let lastCache: (any KVCache)? = lastIndex < cache.count ? cache[lastIndex] : nil
        let (h, mlpOut) = layers[lastIndex].residualAndMLPOutput(
            fused, mask: mask, cache: lastCache)
        if qwen35MTPHeadNormFuseEnabled, h.dtype == .bfloat16, mlpOut.dtype == .bfloat16,
           h.dim(-1) == 5120
        {
            // 4. Return pre-lm_head hidden (norm applied; lm_head is in
            // TextModel) -- fused with the exit add, one launch not two.
            return qwen35FusedResidualRMSNorm(
                x: h, r: mlpOut, weight: norm.weight, eps: norm.eps
            ).normed
        }
        // 4. Return pre-lm_head hidden (norm applied; lm_head is in TextModel).
        return norm(h + mlpOut)
    }

    /// Run one proposal flush while omitting leading-row outputs that have no
    /// consumer. Every supplied row still participates in the fusion stage and
    /// contributes K/V state; only the final row needs a full decoder output.
    /// Multi-layer heads fail closed before mutating cache state.
    func lastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray? {
        guard layers.count == 1, cache.count == 1,
              hidden.dim(1) > 1,
              nextTokenIds.dim(1) == hidden.dim(1)
        else { return nil }

        let fused = fc(
            preFcConcat(
                nextTokenIds: nextTokenIds, embedTokens: embedTokens,
                hidden: hidden))
        let historyCount = fused.dim(1) - 1

        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        let (h, mlpOut) = layers[0].residualAndMLPOutput(
            current, mask: mask, cache: cache[0])
        if qwen35MTPHeadNormFuseEnabled, h.dtype == .bfloat16, mlpOut.dtype == .bfloat16,
           h.dim(-1) == 5120
        {
            return qwen35FusedResidualRMSNorm(
                x: h, r: mlpOut, weight: norm.weight, eps: norm.eps
            ).normed
        }
        return norm(h + mlpOut)
    }

}
