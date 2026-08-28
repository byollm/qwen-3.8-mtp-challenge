// CompiledDecode: whole-step compiled decode helper.
//
// Ported (trimmed) from osaurus-ai/vmlx-swift-lm
// (Libraries/MLXLMCommon/BatchEngine/BatchCompile.swift `compileForward`).
//
// This wraps MLX `compile(inputs:outputs:)` so a single decode step — the model
// forward plus the per-layer KV-cache write — is captured as one compiled graph,
// collapsing hundreds of FFI crossings into a single compiled call. The tracer
// captures each cache layer's `innerState()`; subsequent invocations mutate the
// captured cache objects in place via `_updateInternal`.
//
// REQUIREMENTS / CONSTRAINTS:
// - Every layer must be a ``CompilableKVCache`` or ``CompilableRotatingKVCache``
//   (fixed-shape, MLXArray offset). Standard `KVCacheSimple` / `RotatingKVCache`
//   change state shape per step and cannot be compile-traced.
// - Mixed caches (e.g. Gemma4: KVCacheSimple for full-attention layers,
//   RotatingKVCache for sliding-window layers) are supported via per-layer
//   promotion in `setupCompiledDecode`.
// - The trace specialises on the token-batch shape it first sees (typically
//   `[B, 1]`). A changing batch size forces a recompile, so the batched decode
//   path needs fixed-size buckets (see the port plan for `GenerationBatch`).
//
// This helper is dependency-free w.r.t. the continuous-batching engine: it can
// be exercised in isolation (see CompilableKVCacheTests) and reused by either a
// single-stream or a batched decode loop once the cache-promotion + bucketing
// plumbing lands.

import Foundation
import MLX
import os

private let compiledDecodeLog = Logger(subsystem: "darkbloom", category: "CompiledDecode")

public enum CompiledDecode {

    /// Compiled decode is ON by default. Set `DARKBLOOM_COMPILED_DECODE=0`
    /// to disable. Guards in GenerationBatch ensure it only activates for
    /// B=1 solo decode with supported cache types (no MTP, no SSM).
    public static let isEnabled: Bool = {
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_COMPILED_DECODE"] {
            return !["0", "false", "no", "off"].contains(raw.lowercased())
        }
        return true
    }()

    /// True iff every layer is a compilable cache type and thus
    /// compile-traceable by ``compileForward(model:cacheRef:)``.
    /// Accepts both single-stream (CompilableKVCache, CompilableRotatingKVCache)
    /// and batched (CompilableBatchKVCache, CompilableBatchRotatingKVCache) types.
    public static func eligible(_ cache: [KVCache]) -> Bool {
        !cache.isEmpty && cache.allSatisfy {
            $0 is CompilableKVCache || $0 is CompilableRotatingKVCache
                || $0 is CompilableBatchKVCache || $0 is CompilableBatchRotatingKVCache
        }
    }

    /// Build a compiled forward closure for a decode step.
    ///
    /// The returned closure accepts `[tokens]` (a single `[B, L]` int token
    /// array wrapped in a one-element array) and returns `[logits]` (a single
    /// `[B, L, V]` array). The captured cache layers are mutated in place.
    ///
    /// Supports mixed cache types: single-stream (``CompilableKVCache``,
    /// ``CompilableRotatingKVCache``) and batched (``CompilableBatchKVCache``,
    /// ``CompilableBatchRotatingKVCache``) all expose `innerState()` returning
    /// MLXArrays tracked by `compile(inputs:outputs:)`.
    ///
    /// - Precondition: `cacheRef` is non-empty and every element is a
    ///   compilable cache (see ``eligible(_:)``). Call `eval(cacheRef)`
    ///   before this so no pending tracer ops corrupt state identity.
    ///
    /// - Parameters:
    ///   - model: The language model to trace through.
    ///   - cacheRef: Per-layer compilable cache instances. Captured by the
    ///     returned closure; must not be empty.
    /// - Returns: A `@Sendable` closure mapping `[tokens]` -> `[logits]`.
    public static func compileForward(
        model: any LanguageModel,
        cacheRef: [KVCache]
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        precondition(
            eligible(cacheRef),
            "CompiledDecode.compileForward requires a non-empty cache where every "
                + "layer is a compilable type (single-stream or batched).")

        let capturedModel = model
        let captured = cacheRef

        return compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0]),
                cache: captured.isEmpty ? nil : captured,
                state: nil
            )
            return [result.logits]
        }
    }

    /// Attempt to set up compiled decode for a model + cache pair.
    ///
    /// This converts eligible cache layers to their compilable equivalents
    /// and builds a compiled forward closure. Per-layer promotion handles
    /// heterogeneous caches (e.g. Gemma4 with mixed KVCacheSimple +
    /// RotatingKVCache layers).
    ///
    /// The conversion is only performed when ALL of these conditions hold:
    /// - `DARKBLOOM_COMPILED_DECODE=1` env var is set
    /// - `MLXHardwareInfo.isCompiledDecodeSupported` is true
    /// - Every cache layer is either `KVCacheSimple` or `RotatingKVCache`
    ///
    /// - Parameters:
    ///   - model: The language model.
    ///   - cache: Mutable cache array. On success, entries are replaced with
    ///     their compilable equivalents in place.
    ///   - maxCacheLength: Maximum sequence length for the compiled cache buffers
    ///     (applies to CompilableKVCache; RotatingKVCache uses its own maxCacheSize).
    /// - Returns: A compiled forward closure, or `nil` if setup was skipped.
    public static func setupCompiledDecode(
        model: any LanguageModel,
        cache: inout [KVCache],
        maxCacheLength: Int = 4096
    ) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
        guard isEnabled else { return nil }
        guard MLXHardwareInfo.isCompiledDecodeSupported else {
            compiledDecodeLog.info("Compiled decode skipped: hardware not supported")
            return nil
        }

        // Validate all layers are promotable before doing any conversion.
        for layer in cache {
            if !(layer is KVCacheSimple) && !(layer is RotatingKVCache) {
                compiledDecodeLog.info(
                    "Compiled decode skipped: unsupported cache type (\(type(of: layer)))")
                return nil
            }
        }

        // Materialize all pending cache operations before conversion.
        eval(cache)

        // Per-layer promotion: each layer type gets its compilable equivalent.
        var simpleCount = 0
        var rotatingCount = 0
        for i in 0..<cache.count {
            if let rotating = cache[i] as? RotatingKVCache {
                // RotatingKVCache → CompilableRotatingKVCache
                // (uses its own maxCacheSize, not maxCacheLength)
                cache[i] = CompilableRotatingKVCache.promote(from: rotating, maxLength: maxCacheLength)
                rotatingCount += 1
            } else if let simple = cache[i] as? KVCacheSimple {
                // KVCacheSimple → CompilableKVCache
                cache[i] = CompilableKVCache.promote(from: simple, maxLength: maxCacheLength)
                simpleCount += 1
            }
        }

        // Materialize the new compilable cache buffers
        eval(cache)

        let layerCount = cache.count
        compiledDecodeLog.info(
            "Compiled decode enabled: \(layerCount) layers (\(simpleCount) simple + \(rotatingCount) rotating), maxLength=\(maxCacheLength)")

        return compileForward(model: model, cacheRef: cache)
    }

    /// Set up compiled decode for batched caches (B >= 1).
    ///
    /// Promotes ``BatchKVCache`` layers to ``CompilableBatchKVCache`` and
    /// ``BatchRotatingKVCache`` layers to ``CompilableBatchRotatingKVCache``
    /// in place. Layers that are already compilable are kept as-is.
    ///
    /// ArraysCache / MambaCache layers are unsupported — if any are present,
    /// setup is skipped and `nil` is returned.
    ///
    /// - Parameters:
    ///   - model: The language model.
    ///   - cache: Mutable batched-cache array. On success, entries are replaced
    ///     with their compilable equivalents in place.
    ///   - maxCacheLength: Maximum sequence length for full-attention layers.
    ///     Sliding-window layers use their own maxCacheSize.
    /// - Returns: A compiled forward closure, or `nil` if setup was skipped.
    public static func setupBatchCompiledDecode(
        model: any LanguageModel,
        cache: inout [any BatchedCache],
        maxCacheLength: Int = 4096
    ) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
        guard isEnabled else { return nil }
        guard MLXHardwareInfo.isCompiledDecodeSupported else {
            compiledDecodeLog.info("Batch compiled decode skipped: hardware not supported")
            return nil
        }

        // Validate all layers are promotable.
        for layer in cache {
            if layer is CompilableBatchKVCache || layer is CompilableBatchRotatingKVCache {
                continue  // Already compilable
            }
            if !(layer is BatchKVCache) && !(layer is BatchRotatingKVCache) {
                compiledDecodeLog.info(
                    "Batch compiled decode skipped: unsupported cache type (\(type(of: layer)))")
                return nil
            }
        }

        // Materialize all pending cache operations before conversion.
        eval(cache)

        // Per-layer promotion.
        var fullCount = 0
        var rotatingCount = 0
        for i in 0..<cache.count {
            if cache[i] is CompilableBatchKVCache || cache[i] is CompilableBatchRotatingKVCache {
                // Already compilable — count it.
                if cache[i] is CompilableBatchKVCache { fullCount += 1 }
                else { rotatingCount += 1 }
                continue
            }

            if let rotating = cache[i] as? BatchRotatingKVCache {
                cache[i] = CompilableBatchRotatingKVCache.promote(
                    from: rotating, maxLength: maxCacheLength)
                rotatingCount += 1
            } else if let full = cache[i] as? BatchKVCache {
                cache[i] = CompilableBatchKVCache.promote(
                    from: full, maxLength: maxCacheLength)
                fullCount += 1
            }
        }

        // Materialize the new compilable cache buffers.
        eval(cache)

        let layerCount = cache.count
        compiledDecodeLog.info(
            "Batch compiled decode enabled: \(layerCount) layers (\(fullCount) full + \(rotatingCount) rotating), maxLength=\(maxCacheLength)")

        // Build compiled forward with the caches cast to [KVCache].
        let cacheRef = cache.map { $0 as any KVCache }
        return compileForward(model: model, cacheRef: cacheRef)
    }

    // MARK: - Static-shape verify compile (ZAC-SDL)

    /// A compile-once, static-shape verify graph for the MTP draft loop.
    ///
    /// The hot path dispatches a single verified forward at a fixed maximum
    /// width `1 + maxDepth` once per round. The graph is traced ONCE outside
    /// the timed window, then reused for every round at every legal width
    /// `<= 1 + maxDepth` -- unused trailing positions are filled with the
    /// primary token so their logits are the same value under the verify
    /// recurrence and their contributions are ignored on the accept walk.
    ///
    /// The intent is the Zero-Alloc Compiled Static-Shape MTP Draft Loop
    /// (ZAC-SDL): pre-allocate the verify token buffer, the head-history
    /// scratch and (when consumed) the logits column once per session, and
    /// pay zero host graph build per step.
    ///
    /// The returned closure accepts a single `[1, maxWidth]` int32 token
    /// array whose contents ARE the verify input; the buffer can be the
    /// pre-allocated one if the caller wants a true zero-alloc hot path, in
    /// which case they only need to fill the leading positions each round.
    /// The closure returns `(logits, hidden, normed?)`, all as device arrays,
    /// with `normed` published only when the model surface supports it
    /// (mirroring the existing `callWithHiddenAndNormed` contract).
    public typealias StaticShapeVerify = @Sendable (MLXArray) -> (
        logits: MLXArray, hidden: MLXArray, normed: MLXArray?
    )

    /// Build a compile-once, static-shape verify forward closure. See
    /// ``StaticShapeVerify`` for the input/output contract.
    ///
    /// The closure is for the MTP draft loop's verify forward (a multi-row
    /// target call at `nConfirmed == 1`); the cache layout must therefore
    /// match the target backbone's verify path. The session that owns the
    /// call site is expected to maintain the cache and token buffer across
    /// rounds so the compile-trace in-place mutation hooks into the same
    /// captured graph every round.
    ///
    /// - Parameters:
    ///   - cacheRef: Per-layer cache. Must be the same array across calls so
    ///     MLX's compile-trace reuses the captured graph in place. Each
    ///     element must implement ``Updatable``/`innerState()` (every cache
    ///     type in the vendored library already does).
    ///   - maxWidth: Static width the compile trace specialises on. The
    ///     verify input is always `[1, maxWidth]`; shorter rounds fill the
    ///     leading positions and let the trailing positions be processed
    ///     for free (the verify forward is weight-stream bound, so the
    ///     extra rows add QMV volume but not host graph build).
    ///   - lmHeadForward: Adapter that calls the model's
    ///     `callWithHiddenAndNormed`-equivalent and returns the
    ///     `(logits, hidden, normed?)` triple. The triple becomes the
    ///     static-shape verify's outer return value; the compile runtime
    ///     itself sees a flat `[MLXArray]` where the third slot is empty
    ///     when the adapter returned `normed == nil`.
    ///   - tokenBuffer: Pre-allocated `[1, maxWidth]` int32 array, kept by
    ///     the caller across rounds. The traced closure reads from it; the
    ///     caller writes new token values into it each round.
    /// - Returns: A `StaticShapeVerify` closure that reads the same
    ///   `tokenBuffer` the caller provided at trace time.
    public static func compileStaticShapeVerify(
        cacheRef: [KVCache],
        maxWidth: Int,
        lmHeadForward: @escaping @Sendable (LMInput.Text) -> (
            MLXArray, MLXArray, MLXArray?
        ),
        tokenBuffer: MLXArray
    ) -> StaticShapeVerify {
        precondition(maxWidth >= 1, "compileStaticShapeVerify: maxWidth must be >= 1")
        precondition(
            tokenBuffer.ndim == 2,
            "compileStaticShapeVerify: tokenBuffer must be 2-D [1, W]")
        precondition(
            tokenBuffer.dim(0) == 1 && tokenBuffer.dim(1) == maxWidth,
            "compileStaticShapeVerify: tokenBuffer shape [\(tokenBuffer.shape)] "
                + "must be [1, \(maxWidth)]")
        precondition(
            tokenBuffer.dtype == .int32,
            "compileStaticShapeVerify: tokenBuffer must be int32")
        precondition(
            !cacheRef.isEmpty,
            "compileStaticShapeVerify: cacheRef must be non-empty")

        let captured = cacheRef
        let forward = lmHeadForward

        // The trace function f is invoked by the compile runtime on the
        // FIRST call only, with `inputs[0]` replaced by the caller-supplied
        // token buffer and the cache's `innerState()` replaced by tracer
        // arrays. Subsequent calls reuse the cached compiled graph; the
        // inputs and outputs are updated in place by the runtime, so the
        // cache the caller passed in keeps mutating through the same
        // captured object.
        //
        // The compile runtime can only return a flat `[MLXArray]`, so the
        // `normed == nil` case ships an empty slot. The outer wrapper
        // below unwraps the flat array into the documented tuple.
        let compiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(
            inputs: [tokenBuffer] + captured,
            outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            // args[0] is the tracer view of `tokenBuffer`; the upstream
            // call site fills the real buffer with [primary, draft_0, ...
            // draft_{D-1}, primary, primary, ...] each round.
            let (logits, hidden, normed) = forward(
                LMInput.Text(tokens: args[0]))
            if let normed { return [logits, hidden, normed] }
            return [logits, hidden]
        }

        return { tokenBuffer in
            let outs = compiled([tokenBuffer])
            // outs is always [logits, hidden] or [logits, hidden, normed];
            // the adapter above controls which.
            if outs.count >= 3 {
                return (outs[0], outs[1], outs[2])
            }
            return (outs[0], outs[1], nil)
        }
    }
}
