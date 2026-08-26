import Foundation
import MLX
import MLXNN

final class Qwen38DFlash2Runtime {
    private let model: Qwen38DraftModel

    init(headDirectory: URL) throws {
        let model = Qwen38DraftModel()
        var weights = try MLX.loadArrays(
            url: headDirectory.appendingPathComponent("model.safetensors")
        )
        if let value = weights.removeValue(
            forKey: "candidate_selector.predecessor_codebook"
        ) {
            weights["candidate_selector.predecessor_codebook.weight"] = value
        }
        if let value = weights.removeValue(
            forKey: "candidate_selector.successor_codebook"
        ) {
            weights["candidate_selector.successor_codebook.weight"] = value
        }

        var packedArrays = [MLXArray]()
        let replacements = model.leafModules().flattened().compactMap {
            path, module -> (String, Module)? in
            guard module is Linear,
                  let weight = weights["\(path).weight"],
                  let scales = weights["\(path).scales"],
                  let biases = weights["\(path).biases"]
            else { return nil }
            weights["\(path).weight"] = nil
            weights["\(path).scales"] = nil
            weights["\(path).biases"] = nil
            let replacement = Qwen38DFlash2Linear(
                weight: weight,
                scales: scales,
                biases: biases
            )
            packedArrays.append(contentsOf: replacement.packedArrays)
            return (path, replacement)
        }
        model.update(modules: ModuleChildren.unflattened(replacements))
        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.noUnusedKeys, .shapeMismatch]
        )
        eval(model, packedArrays)
        self.model = model
    }

    func makeState() -> Qwen38DraftState {
        model.makeState()
    }

    func propose(
        anchor: MLXArray,
        taps: MLXArray,
        target: any Qwen36MTPTarget,
        state: inout Qwen38DraftState
    ) -> MLXArray {
        model.propose(
            anchor: anchor,
            targetTaps: taps,
            embedding: { target.dFlash2Embed($0) },
            head: { target.applyLMHead($0) },
            state: &state
        )
    }
}
