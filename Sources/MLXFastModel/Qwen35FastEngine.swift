import MLX
import MLXFastCore
import MLXLMCommon

/// Editable custom text trunk. It deliberately stops at normalized hidden
/// states so vocabulary projection policy is an independent seam.
public struct Qwen35FastTrunk {
    public let embedding: Qwen35LinearWeight
    public let blocks: [Qwen35BlockWeights]
    public let finalNorm: MLXArray
    public let hiddenSize: Int
    public let rmsNormEps: Double

    public init(
        embedding: Qwen35LinearWeight,
        blocks: [Qwen35BlockWeights],
        finalNorm: MLXArray,
        hiddenSize: Int,
        rmsNormEps: Double
    ) throws {
        guard hiddenSize > 0,
              !blocks.isEmpty,
              embedding.shape.count == 2,
              embedding.shape[1] == hiddenSize,
              finalNorm.shape == [hiddenSize]
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 fast trunk dimensions are invalid"
            )
        }
        self.embedding = embedding
        self.blocks = blocks
        self.finalNorm = finalNorm
        self.hiddenSize = hiddenSize
        self.rmsNormEps = rmsNormEps
        try validateContract()
    }

    public var layerTypes: [Qwen35LayerType] {
        blocks.map(\.mixer.layerType)
    }

    public func callAsFunction(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache? = nil,
        positionOffset: Int = 0,
        ssmMask: MLXArray? = nil
    ) throws -> MLXArray {
        try forward(
            inputIDs,
            cache: cache,
            positionOffset: positionOffset,
            ssmMask: ssmMask,
            afterLayer: nil
        )
    }

    func forward(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache?,
        positionOffset: Int,
        ssmMask: MLXArray?,
        afterLayer: ((Int) throws -> Void)?
    ) throws -> MLXArray {
        guard inputIDs.ndim == 2,
              inputIDs.dim(0) > 0,
              inputIDs.dim(1) > 0,
              positionOffset >= 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 fast trunk input IDs or position are invalid"
            )
        }
        if let ssmMask {
            guard ssmMask.shape == [inputIDs.dim(0), inputIDs.dim(1)],
                  ssmMask.dtype == .bool
            else {
                throw MLXFastError.invalidInput(
                    "Qwen35 fast trunk SSM mask must be boolean [batch, length]"
                )
            }
        }
        if cache == nil, positionOffset != 0 {
            throw MLXFastError.invalidInput(
                "Qwen35 fast trunk requires cache state for nonzero positions"
            )
        }

        let nextPosition = try cache?.validateAdvance(
            positionOffset: positionOffset,
            inputLength: inputIDs.dim(1),
            expectedLayerCount: blocks.count
        )
        var hidden = Qwen35Ops.embedding(
            inputIDs: inputIDs,
            weight: embedding
        )

        for (index, block) in blocks.enumerated() {
            let layerCache = cache?.layers[index]
            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            switch block.mixer {
            case .linear:
                attentionMask = .none
            case .full:
                attentionMask = makeAttentionMask(
                    n: inputIDs.dim(1),
                    cache: layerCache?.fullAttention
                )
            }
            hidden = try Qwen35Block.forward(
                hidden,
                weights: block,
                rmsNormEps: rmsNormEps,
                attentionMask: attentionMask,
                ssmMask: ssmMask,
                cache: layerCache,
                positionOffset: positionOffset
            )
            try afterLayer?(index)
        }

        let normalized = Qwen35Ops.rmsNorm(
            hidden,
            weight: finalNorm,
            eps: rmsNormEps
        )
        if let nextPosition {
            cache?.commit(positionOffset: nextPosition)
        }
        return normalized
    }

    /// VSRS-PHR (Verify-Side Parent-Hidden-State Reuse Ring): when a
    /// `Qwen35ParentHiddenRing` is attached, capture the post-norm hidden for
    /// the LAST position of this forward so the draft path can read it back
    /// instead of re-running embedding gather + RoPE + the parent block.
    func forwardCapturingParent(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache?,
        positionOffset: Int,
        ssmMask: MLXArray?,
        ring: Qwen35ParentHiddenRing?
    ) throws -> MLXArray {
        let normalized = try forward(
            inputIDs,
            cache: cache,
            positionOffset: positionOffset,
            ssmMask: ssmMask,
            afterLayer: nil
        )
        if let ring {
            let lastToken = normalized.dim(1)
            if lastToken > 0 {
                ring.record(
                    absolutePosition: positionOffset + lastToken - 1,
                    hidden: hiddenRow(normalized, lastToken - 1)
                )
            }
        }
        return normalized
    }

    private func hiddenRow(_ hidden: MLXArray, _ row: Int) -> MLXArray {
        let last = hidden.dim(1) - 1
        if row == last, row == 0 {
            return hidden
        }
        return hidden[0..., row ..< (row + 1), 0...]
    }

    func validateContract() throws {
        try embedding.validate()
        guard hiddenSize > 0,
              !blocks.isEmpty,
              embedding.shape.count == 2,
              embedding.shape[1] == hiddenSize,
              finalNorm.shape == [hiddenSize],
              rmsNormEps.isFinite,
              rmsNormEps > 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 fast trunk contract is invalid"
            )
        }
        for block in blocks {
            try Qwen35Block.validateContract(
                weights: block,
                hiddenSize: hiddenSize
            )
        }
    }
}

/// Hybrid cache for the custom trunk: ordinary public KV caches for full
/// attention and package-owned states for Gated DeltaNet.
public final class Qwen35FastCache {
    public let layers: [Qwen35BlockCache]
    public private(set) var expectedPositionOffset = 0

    public init(trunk: Qwen35FastTrunk) {
        self.layers = trunk.blocks.map { block in
            switch block.mixer {
            case .linear:
                Qwen35BlockCache()
            case .full:
                Qwen35BlockCache(fullAttention: KVCacheSimple())
            }
        }
    }

    func validateAdvance(
        positionOffset: Int,
        inputLength: Int,
        expectedLayerCount: Int
    ) throws -> Int {
        guard layers.count == expectedLayerCount,
              positionOffset == expectedPositionOffset,
              inputLength > 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 fast cache position or topology is invalid"
            )
        }
        let next = expectedPositionOffset.addingReportingOverflow(
            inputLength
        )
        guard !next.overflow else {
            throw MLXFastError.invalidInput(
                "Qwen35 fast cache position overflows Int"
            )
        }
        for layer in layers {
            if let fullAttention = layer.fullAttention,
               fullAttention.offset != positionOffset
            {
                throw MLXFastError.invalidInput(
                    "Qwen35 fast full-attention cache offset "
                        + "\(fullAttention.offset) does not match "
                        + "\(positionOffset)"
                )
            }
        }
        return next.partialValue
    }

    func commit(positionOffset: Int) {
        expectedPositionOffset = positionOffset
    }

    func withRollback<Result>(
        _ body: () throws -> Result
    ) throws -> Result {
        let snapshot = Qwen35FastCacheSnapshot(
            expectedPositionOffset: expectedPositionOffset,
            layers: layers.map { $0.snapshot() }
        )
        do {
            return try body()
        } catch {
            expectedPositionOffset = snapshot.expectedPositionOffset
            for (layer, saved) in zip(layers, snapshot.layers) {
                layer.restore(saved)
            }
            throw error
        }
    }

    public func materialize() {
        var arrays: [MLXArray] = []
        for layer in layers {
            if let fullAttention = layer.fullAttention {
                arrays.append(contentsOf: fullAttention.state)
            }
            if let gatedDelta = layer.gatedDelta {
                arrays.append(gatedDelta.convolution)
                arrays.append(gatedDelta.recurrent)
            }
        }
        eval(arrays)
    }
}

private struct Qwen35FastCacheSnapshot {
    let expectedPositionOffset: Int
    let layers: [Qwen35BlockCacheSnapshot]
}

/// Explicit untied output head. `lastTokenLogits` slices hidden states before
/// the 248,320-way projection, allowing a future scored prefill path to avoid
/// materializing all-position logits without changing the trunk.
public struct Qwen35LMHead {
    public let weight: Qwen35LinearWeight

    public init(weight: Qwen35LinearWeight) {
        self.weight = weight
    }

    public func logits(_ normalizedHidden: MLXArray) throws -> MLXArray {
        try validate(normalizedHidden)
        return Qwen35Ops.linear(normalizedHidden, weight)
    }

    public func lastTokenLogits(
        _ normalizedHidden: MLXArray
    ) throws -> MLXArray {
        try validate(normalizedHidden)
        let last = normalizedHidden[
            0...,
            (normalizedHidden.dim(1) - 1)..<normalizedHidden.dim(1),
            0...
        ]
        return Qwen35Ops.linear(last, weight)
    }

    private func validate(_ normalizedHidden: MLXArray) throws {
        guard normalizedHidden.ndim == 3,
              normalizedHidden.dim(1) > 0
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 LM-head input or weight shape is invalid"
            )
        }
        try validateContract(hiddenSize: normalizedHidden.dim(2))
    }

    func validateContract(hiddenSize: Int) throws {
        try weight.validate()
        guard weight.shape.count == 2,
              weight.shape[0] > 0,
              weight.shape[1] == hiddenSize
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 LM-head weight shape is invalid"
            )
        }
    }
}

enum Qwen35FastFailurePoint: Equatable {
    case afterLayer(Int)
    case afterHead
}

private enum Qwen35FastInjectedFailure: Error {
    case requested
}

private enum Qwen35HeadProjection {
    case allPositions
    case lastToken
}

public struct Qwen35FastEngine {
    public let trunk: Qwen35FastTrunk
    public let lmHead: Qwen35LMHead

    public init(
        trunk: Qwen35FastTrunk,
        lmHead: Qwen35LMHead
    ) throws {
        self.trunk = trunk
        self.lmHead = lmHead
        try validateContract()
    }

    public func newCache() -> Qwen35FastCache {
        Qwen35FastCache(trunk: trunk)
    }

    public func allPositionLogits(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache? = nil,
        positionOffset: Int = 0,
        ssmMask: MLXArray? = nil
    ) throws -> MLXArray {
        try forward(
            inputIDs,
            cache: cache,
            positionOffset: positionOffset,
            ssmMask: ssmMask,
            projection: .allPositions,
            failurePoint: nil
        )
    }

    public func lastTokenLogits(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache? = nil,
        positionOffset: Int = 0,
        ssmMask: MLXArray? = nil
    ) throws -> MLXArray {
        try forward(
            inputIDs,
            cache: cache,
            positionOffset: positionOffset,
            ssmMask: ssmMask,
            projection: .lastToken,
            failurePoint: nil
        )
    }

    func allPositionLogits(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache,
        positionOffset: Int,
        ssmMask: MLXArray? = nil,
        failurePoint: Qwen35FastFailurePoint
    ) throws -> MLXArray {
        try forward(
            inputIDs,
            cache: cache,
            positionOffset: positionOffset,
            ssmMask: ssmMask,
            projection: .allPositions,
            failurePoint: failurePoint
        )
    }

    private func forward(
        _ inputIDs: MLXArray,
        cache: Qwen35FastCache?,
        positionOffset: Int,
        ssmMask: MLXArray?,
        projection: Qwen35HeadProjection,
        failurePoint: Qwen35FastFailurePoint?
    ) throws -> MLXArray {
        // Validate the complete immutable graph before any layer can mutate a
        // full-attention or recurrent cache.
        try validateContract()
        let activeCache = cache ?? newCache()
        return try activeCache.withRollback {
            let hidden = try trunk.forward(
                inputIDs,
                cache: activeCache,
                positionOffset: positionOffset,
                ssmMask: ssmMask,
                afterLayer: { layerIndex in
                    if failurePoint == .afterLayer(layerIndex) {
                        throw Qwen35FastInjectedFailure.requested
                    }
                }
            )
            let logits: MLXArray
            switch projection {
            case .allPositions:
                logits = try lmHead.logits(hidden)
            case .lastToken:
                logits = try lmHead.lastTokenLogits(hidden)
            }
            if failurePoint == .afterHead {
                throw Qwen35FastInjectedFailure.requested
            }
            return logits
        }
    }

    private func validateContract() throws {
        try trunk.validateContract()
        try lmHead.validateContract(hiddenSize: trunk.hiddenSize)
    }

    /// Materializes a complete custom engine from the transformed checkpoint.
    /// Production does not call this while the readiness gate is closed.
    public static func load(
        loader: Qwen35WeightLoader,
        config: Qwen35Config
    ) throws -> Qwen35FastEngine {
        try config.validateFrozenInvariants()
        try config.validateStructuralValues()
        try loader.denseStore.validateReadableByteRanges()
        try loader.validateRequiredMetadata(config: config)

        func linear(
            _ name: String,
            output: Int,
            input: Int
        ) throws -> Qwen35LinearWeight {
            try loader.linearWeight(
                named: name,
                outFeatures: output,
                inFeatures: input,
                expectedGroupSize: config.quantizationGroupSize,
                expectedBits: config.quantizationBits,
                requireQuantized: true
            )
        }

        let embedding = try linear(
            Qwen35WeightNames.embedTokens,
            output: config.vocabSize,
            input: config.hiddenSize
        )
        let attentionSpec = try Qwen35AttentionSpec(config: config)
        let gatedDeltaSpec = try Qwen35GatedDeltaSpec(config: config)
        var blocks: [Qwen35BlockWeights] = []
        blocks.reserveCapacity(config.numHiddenLayers)

        for layerIndex in 0..<config.numHiddenLayers {
            let mlp = Qwen35MLPWeights(
                gateProjection: try linear(
                    Qwen35WeightNames.mlp(
                        layerIndex,
                        "gate_proj.weight"
                    ),
                    output: config.intermediateSize,
                    input: config.hiddenSize
                ),
                upProjection: try linear(
                    Qwen35WeightNames.mlp(
                        layerIndex,
                        "up_proj.weight"
                    ),
                    output: config.intermediateSize,
                    input: config.hiddenSize
                ),
                downProjection: try linear(
                    Qwen35WeightNames.mlp(
                        layerIndex,
                        "down_proj.weight"
                    ),
                    output: config.hiddenSize,
                    input: config.intermediateSize
                )
            )

            let mixer: Qwen35BlockMixer
            switch config.layerTypes[layerIndex] {
            case .linear:
                let prefix: (String) -> String = {
                    Qwen35WeightNames.linearAttention(layerIndex, $0)
                }
                mixer = .linear(
                    weights: Qwen35GatedDeltaWeights(
                        inputQKVProjection: try linear(
                            prefix("in_proj_qkv.weight"),
                            output: gatedDeltaSpec.convolutionDimension,
                            input: config.hiddenSize
                        ),
                        inputZProjection: try linear(
                            prefix("in_proj_z.weight"),
                            output: gatedDeltaSpec.valueSize,
                            input: config.hiddenSize
                        ),
                        inputBProjection: try linear(
                            prefix("in_proj_b.weight"),
                            output: gatedDeltaSpec.numValueHeads,
                            input: config.hiddenSize
                        ),
                        inputAProjection: try linear(
                            prefix("in_proj_a.weight"),
                            output: gatedDeltaSpec.numValueHeads,
                            input: config.hiddenSize
                        ),
                        convolution: try loader.denseArray(
                            named: prefix("conv1d.weight"),
                            expectedShape: [
                                gatedDeltaSpec.convolutionDimension,
                                gatedDeltaSpec.convolutionKernelSize,
                                1,
                            ]
                        ),
                        timeStepBias: try loader.denseArray(
                            named: prefix("dt_bias"),
                            expectedShape: [
                                gatedDeltaSpec.numValueHeads
                            ]
                        ),
                        aLog: try loader.denseArray(
                            named: prefix("A_log"),
                            expectedShape: [
                                gatedDeltaSpec.numValueHeads
                            ]
                        ),
                        outputNorm: try loader.denseArray(
                            named: prefix("norm.weight"),
                            expectedShape: [
                                gatedDeltaSpec.valueHeadDimension
                            ]
                        ),
                        outputProjection: try linear(
                            prefix("out_proj.weight"),
                            output: config.hiddenSize,
                            input: gatedDeltaSpec.valueSize
                        )
                    ),
                    spec: gatedDeltaSpec
                )

            case .full:
                let prefix: (String) -> String = {
                    Qwen35WeightNames.fullAttention(layerIndex, $0)
                }
                mixer = .full(
                    weights: Qwen35AttentionWeights(
                        queryProjection: try linear(
                            prefix("q_proj.weight"),
                            output: attentionSpec
                                .queryAndGateProjectionSize,
                            input: config.hiddenSize
                        ),
                        keyProjection: try linear(
                            prefix("k_proj.weight"),
                            output: attentionSpec.keyValueSize,
                            input: config.hiddenSize
                        ),
                        valueProjection: try linear(
                            prefix("v_proj.weight"),
                            output: attentionSpec.keyValueSize,
                            input: config.hiddenSize
                        ),
                        outputProjection: try linear(
                            prefix("o_proj.weight"),
                            output: config.hiddenSize,
                            input: attentionSpec.querySize
                        ),
                        queryNorm: try loader.denseArray(
                            named: prefix("q_norm.weight"),
                            expectedShape: [config.headDim]
                        ),
                        keyNorm: try loader.denseArray(
                            named: prefix("k_norm.weight"),
                            expectedShape: [config.headDim]
                        )
                    ),
                    spec: attentionSpec
                )
            }

            blocks.append(
                Qwen35BlockWeights(
                    inputLayerNorm: try loader.denseArray(
                        named: Qwen35WeightNames.layer(
                            layerIndex,
                            "input_layernorm.weight"
                        ),
                        expectedShape: [config.hiddenSize]
                    ),
                    postAttentionLayerNorm: try loader.denseArray(
                        named: Qwen35WeightNames.layer(
                            layerIndex,
                            "post_attention_layernorm.weight"
                        ),
                        expectedShape: [config.hiddenSize]
                    ),
                    mixer: mixer,
                    mlp: mlp
                )
            )
        }

        let trunk = try Qwen35FastTrunk(
            embedding: embedding,
            blocks: blocks,
            finalNorm: try loader.denseArray(
                named: Qwen35WeightNames.finalNorm,
                expectedShape: [config.hiddenSize]
            ),
            hiddenSize: config.hiddenSize,
            rmsNormEps: config.rmsNormEps
        )
        return try Qwen35FastEngine(
            trunk: trunk,
            lmHead: Qwen35LMHead(
                weight: try linear(
                    Qwen35WeightNames.lmHead,
                    output: config.vocabSize,
                    input: config.hiddenSize
                )
            )
        )
    }
}

// MARK: - VSRS-PHR (Verify-Side Parent-Hidden-State Reuse Ring)
//
// A small fixed-capacity ring buffer of post-norm trunk hidden states keyed by
// the absolute position the model wrote them at. The verify-side draft path
// reads `parentHiddenAt(position:)` instead of re-running the parent block, the
// embedding gather and the RoPE application; an unfilled slot (or a position
// outside the ring window) returns nil and the caller falls back to the
// original re-derive. The ring is INTERNAL STATE OF THE DECODE, not of the
// model: it is per-worker, per-prompt, and never serialised to the audit
// record. The recorded hidden rows are device-resident MLXArrays — same
// memory as the verify forward's final layer — so a hit is a copy-free lookup
// of an already-evaluated tensor.
public final class Qwen35ParentHiddenRing {
    /// Capacity. The shipped value is small: a few parent-block outputs is
    /// enough for the verify path to amortise the cost of re-deriving the
    /// parent's RoPE'd block over the next few draft positions. Anything
    /// larger eats memory the ranked M5 has reserved for the live caches.
    public let reuseRingSize: Int
    private struct Slot {
        var position: Int
        var hidden: MLXArray?
    }
    private var slots: [Slot]
    private var headIndex: Int = 0
    /// Total successful records (monotonic, survives eviction). Used by the
    /// hit-rate telemetry without exposing the per-slot history.
    public private(set) var recordCount: Int = 0
    public private(set) var hitCount: Int = 0
    public private(set) var missCount: Int = 0

    public init(reuseRingSize: Int) {
        precondition(reuseRingSize > 0, "reuse_ring_size must be positive")
        self.reuseRingSize = reuseRingSize
        self.slots = Array(repeating: Slot(position: -1, hidden: nil),
                           count: reuseRingSize)
    }

    /// Insert (or overwrite) the ring slot for `absolutePosition`. Overwriting
    /// the same position is benign and never evicts a different position.
    public func record(absolutePosition: Int, hidden: MLXArray) {
        if let existing = findSlot(absolutePosition: absolutePosition) {
            slots[existing].hidden = hidden
            recordCount += 1
            return
        }
        slots[headIndex] = Slot(position: absolutePosition, hidden: hidden)
        headIndex = (headIndex + 1) % reuseRingSize
        recordCount += 1
    }

    /// Read the cached hidden state for `absolutePosition`, if the ring still
    /// holds it. Returns nil for any miss (eviction, never recorded, or
    /// outside the window).
    public func parentHiddenAt(absolutePosition: Int) -> MLXArray? {
        guard let index = findSlot(absolutePosition: absolutePosition) else {
            missCount += 1
            return nil
        }
        hitCount += 1
        return slots[index].hidden
    }

    /// Read multiple contiguous positions starting at `startPosition`, oldest
    /// first. Returns one hidden per slot that is in `[startPosition,
    /// startPosition + count)`. Used by the draft path to seed a chain of
    /// head sub-steps in one lookup.
    public func parentHiddenRange(
        startPosition: Int,
        count: Int
    ) -> [MLXArray?] {
        guard count > 0 else { return [] }
        var out: [MLXArray?] = Array(repeating: nil, count: count)
        for offset in 0 ..< count {
            let position = startPosition + offset
            if let index = findSlot(absolutePosition: position) {
                out[offset] = slots[index].hidden
            }
        }
        return out
    }

    public func clear() {
        for index in slots.indices {
            slots[index] = Slot(position: -1, hidden: nil)
        }
        headIndex = 0
        recordCount = 0
        hitCount = 0
        missCount = 0
    }

    public var occupancy: Int {
        slots.reduce(0) { $0 + ($1.hidden == nil ? 0 : 1) }
    }

    public var hitRate: Double {
        let total = hitCount + missCount
        guard total > 0 else { return 0.0 }
        return Double(hitCount) / Double(total)
    }

    private func findSlot(absolutePosition: Int) -> Int? {
        for (index, slot) in slots.enumerated() where slot.position == absolutePosition {
            return index
        }
        return nil
    }
}

// MARK: - Acceptance-EMA-driven depth backoff selector
//
// Maps the current acceptance-rate EMA into one of three discrete draft-depth
// backoff levels via two thresholds. A candidate `currentEMA` above the high
// threshold returns `depthBackoffHigh` (no backoff, full offered depth), an
// EMA below the low threshold returns `depthBackoffLow` (a deep backoff that
// may even collapse to a serial control), and a mid-band EMA returns
// `depthBackoffMid`. The point is graceful degradation: instead of a hard
// collapse to zero drafts when the head stops accepting, the schedule eases
// off through a stepped ladder so a partial-recovery stretch resumes
// drafting without paying the cold-start ramp from scratch.
public struct Qwen35AcceptanceEMADepthSelector {
    public let depthBackoffHigh: Int
    public let depthBackoffMid: Int
    public let depthBackoffLow: Int
    public let emaHighThreshold: Double
    public let emaLowThreshold: Double
    public let offeredDepth: Int

    public init(
        depthBackoffHigh: Int,
        depthBackoffMid: Int,
        depthBackoffLow: Int,
        emaHighThreshold: Double,
        emaLowThreshold: Double,
        offeredDepth: Int
    ) {
        precondition(depthBackoffHigh >= 0)
        precondition(depthBackoffMid >= 0)
        precondition(depthBackoffLow >= 0)
        precondition(emaHighThreshold > emaLowThreshold)
        precondition(emaHighThreshold <= 1.0)
        precondition(emaLowThreshold >= 0.0)
        precondition(offeredDepth >= 0)
        self.depthBackoffHigh = depthBackoffHigh
        self.depthBackoffMid = depthBackoffMid
        self.depthBackoffLow = depthBackoffLow
        self.emaHighThreshold = emaHighThreshold
        self.emaLowThreshold = emaLowThreshold
        self.offeredDepth = offeredDepth
    }

    /// Map the current EMA into a backoff depth. The output is clipped to
    /// `offeredDepth` so the contract the trusted parent enforces still holds.
    public func selectDepth(currentEMA: Double) -> Int {
        let raw: Int
        if currentEMA >= emaHighThreshold {
            raw = depthBackoffHigh
        } else if currentEMA >= emaLowThreshold {
            raw = depthBackoffMid
        } else {
            raw = depthBackoffLow
        }
        let offered = Swift.max(0, offeredDepth)
        return Swift.min(raw, offered)
    }
}

// MARK: - Engine-side fast-path that USES the ring and the selector
//
// `Qwen35FastEngine` is the editable surface the worker compiles; the two
// primitives above would be dormant without a runtime use. The methods below
// expose a public, observable change: the verify path can now look up a
// cached parent hidden row from the ring, and the depth schedule can now
// consult the EMA-driven backoff selector in addition to (or instead of) the
// shipped cost-model ladder. None of this is wired into the existing
// `generateRound` (which lives in a separate editable file); it lives here so
// any new decode path that builds on `Qwen35FastEngine` can opt in without
// having to re-derive the per-position state.
extension Qwen35FastEngine {
    /// Default ring configuration: capacity 8, large enough to cover the
    /// pinned 2..7 draft window plus a couple of bonus rows, small enough not
    /// to push the M5's 128 GiB envelope.
    public static let defaultReuseRingSize = 8

    /// Build a parent-hidden ring the verify path can populate.
    public func makeParentHiddenRing(
        reuseRingSize: Int = Qwen35FastEngine.defaultReuseRingSize
    ) -> Qwen35ParentHiddenRing {
        Qwen35ParentHiddenRing(reuseRingSize: reuseRingSize)
    }

    /// Forward the engine and capture the last position's post-norm hidden
    /// state into the supplied ring. The returned tensor is the same one the
    /// cached call would have returned, so existing callers stay correct.
    public func lastTokenLogitsIntoRing(
        _ inputIDs: MLXArray,
        ring: Qwen35ParentHiddenRing,
        cache: Qwen35FastCache? = nil,
        positionOffset: Int = 0,
        ssmMask: MLXArray? = nil
    ) throws -> MLXArray {
        let active = cache ?? newCache()
        return try active.withRollback {
            let normalized = try trunk.forwardCapturingParent(
                inputIDs,
                cache: active,
                positionOffset: positionOffset,
                ssmMask: ssmMask,
                ring: ring
            )
            return try lmHead.lastTokenLogits(normalized)
        }
    }

    /// Read a parent hidden row from the ring, or compute it on demand by
    /// running a one-token forward through the engine. The fallback path is
    /// the only one callers can hit when the ring is cold (e.g. the first
    /// position of a new prompt); once the ring has at least one entry, a
    /// hit is a direct MLXArray lookup and the parent block is never
    /// re-evaluated. The cache advances by the on-demand fallback's length
    /// so the contract with the verify forward stays intact.
    public func parentHiddenFor(
        _ position: Int,
        ring: Qwen35ParentHiddenRing,
        cache: Qwen35FastCache,
        fallbackToken: Int
    ) throws -> MLXArray {
        if let cached = ring.parentHiddenAt(absolutePosition: position) {
            return cached
        }
        let normalized = try trunk.forward(
            MLXArray([Int32(fallbackToken)]).reshaped([1, 1]),
            cache: cache,
            positionOffset: position,
            ssmMask: nil,
            afterLayer: nil
        )
        let row = normalized.dim(1) - 1
        let slice = row > 0
            ? normalized[0..., row ..< (row + 1), 0...]
            : normalized
        ring.record(absolutePosition: position, hidden: slice)
        return slice
    }

    /// Build the EMA-driven backoff selector the depth schedule can read.
    public static func makeEMADepthSelector(
        depthBackoffHigh: Int = 4,
        depthBackoffMid: Int = 2,
        depthBackoffLow: Int = 1,
        emaHighThreshold: Double = 0.70,
        emaLowThreshold: Double = 0.40,
        offeredDepth: Int = 4
    ) -> Qwen35AcceptanceEMADepthSelector {
        Qwen35AcceptanceEMADepthSelector(
            depthBackoffHigh: depthBackoffHigh,
            depthBackoffMid: depthBackoffMid,
            depthBackoffLow: depthBackoffLow,
            emaHighThreshold: emaHighThreshold,
            emaLowThreshold: emaLowThreshold,
            offeredDepth: offeredDepth
        )
    }

    /// Apply the EMA-driven backoff on top of the shipped cost-model depth.
    /// The selector is the FAILSAFE: when its computed depth is lower than
    /// the cost-model output, the selector wins; when it is higher, the cost
    /// model keeps its value. That ordering prevents a healthy cost-model
    /// estimate from being throttled by a momentarily noisy EMA, while
    /// still letting the selector protect the run from a sustained cold
    /// stretch. Returning the cost-model output untouched when the selector
    /// is exactly the offer keeps the round byte-identical to the shipped
    /// default on a healthy prompt.
    public func reconcileDepths(
        costModelDepth: Int,
        currentEMA: Double,
        selector: Qwen35AcceptanceEMADepthSelector
    ) -> Int {
        let backoff = selector.selectDepth(currentEMA: currentEMA)
        if backoff >= costModelDepth {
            return costModelDepth
        }
        return backoff
    }
}
