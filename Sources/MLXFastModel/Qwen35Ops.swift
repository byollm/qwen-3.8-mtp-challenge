import MLX
import MLXLMCommon
import MLXNN

/// Architecture-neutral tensor primitives used by the custom Qwen35 path.
public enum Qwen35Ops {
    /// Avoid an otherwise redundant graph node when the dtype already matches.
    @inline(__always)
    public static func cast(_ input: MLXArray, to dtype: DType) -> MLXArray {
        input.dtype == dtype ? input : input.asType(dtype)
    }

    public static func embedding(
        inputIDs: MLXArray,
        weight: Qwen35LinearWeight
    ) -> MLXArray {
        guard let scales = weight.scales else {
            return weight.weight[inputIDs]
        }
        return dequantized(
            weight.weight[inputIDs],
            scales: scales[inputIDs],
            biases: weight.biases.map { $0[inputIDs] },
            groupSize: weight.groupSize,
            bits: weight.bits,
            mode: .affine
        )
    }

    public static func linear(
        _ input: MLXArray,
        _ weight: Qwen35LinearWeight
    ) -> MLXArray {
        guard let scales = weight.scales else {
            return matmul(input, weight.weight.T)
        }
        return quantizedMM(
            input,
            weight.weight,
            scales: scales,
            biases: weight.biases,
            transpose: true,
            groupSize: weight.groupSize,
            bits: weight.bits,
            mode: .affine
        )
    }

    /// Qwen35's learned RMSNorm uses the checkpoint weight directly.
    @inline(__always)
    public static func rmsNorm(
        _ input: MLXArray,
        weight: MLXArray,
        eps: Double
    ) -> MLXArray {
        MLXFast.rmsNorm(input, weight: weight, eps: Float(eps))
    }

    /// Weightless RMSNorm used for Gated DeltaNet Q/K normalization.
    @inline(__always)
    public static func rmsNorm(
        _ input: MLXArray,
        eps: Double
    ) -> MLXArray {
        MLXFast.rmsNorm(
            input,
            weight: MLXArray.mlxNone,
            eps: Float(eps)
        )
    }

    private static let compiledPreciseGateAndMul:
        @Sendable (MLXArray, MLXArray) -> MLXArray =
    {
        let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, normalized in
            (silu(gate.asType(.float32)) * normalized.asType(.float32))
                .asType(normalized.dtype)
        }
        if MLXHardwareInfo.isCompiledDecodeSupported {
            return compile(shapeless: true, body)
        }
        return body
    }()

    /// Precise gated RMSNorm from pinned `Qwen3NextRMSNormGated`:
    /// normalize first, then compute SiLU(gate) and the product in float32.
    /// Fuses the SiLU activation, float32 precision casts, elementwise product,
    /// and output conversion into a single compiled Metal kernel launch.
    public static func preciseGatedRMSNorm(
        _ input: MLXArray,
        gate: MLXArray,
        weight: MLXArray,
        eps: Double
    ) -> MLXArray {
        let normalized = rmsNorm(input, weight: weight, eps: eps)
        return compiledPreciseGateAndMul(gate, normalized)
    }

    private static let compiledSigmoidGate:
        @Sendable (MLXArray, MLXArray) -> MLXArray =
    {
        let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { attended, gate in
            attended * sigmoid(gate)
        }
        if MLXHardwareInfo.isCompiledDecodeSupported {
            return compile(shapeless: true, body)
        }
        return body
    }()

    /// Fuses full-attention output gating `attended * sigmoid(gate)` into a single
    /// compiled elementwise pass, avoiding intermediate allocation for sigmoid(gate).
    public static func sigmoidGate(
        _ attended: MLXArray,
        gate: MLXArray
    ) -> MLXArray {
        compiledSigmoidGate(attended, gate)
    }

    private static let compiledSwiGLU:
        @Sendable (MLXArray, MLXArray) -> MLXArray =
    {
        let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
            silu(gate) * up
        }
        if MLXHardwareInfo.isCompiledDecodeSupported {
            return compile(shapeless: true, body)
        }
        return body
    }()

    /// Fuses SwiGLU activation `silu(gate) * up` into a single compiled elementwise pass.
    public static func swiglu(
        _ gate: MLXArray,
        _ up: MLXArray
    ) -> MLXArray {
        compiledSwiGLU(gate, up)
    }
}
