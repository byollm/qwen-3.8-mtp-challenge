import Foundation

/// Tracks the token sequence a hybrid Qwen3.8 cache represents, and decides
/// whether a new prompt can CONTINUE that cache instead of re-prefilling it.
///
/// WHY THIS EXISTS. The native MTP worker builds a fresh session for every
/// HTTP request, so every request re-prefills the whole prompt. On the pinned
/// model (Qwen3.8-27B: 48 GatedDeltaNet layers + 16 full-attention layers)
/// prefill costs roughly 5 ms/token against roughly 22 ms/token for decode,
/// so a 10.8k-token prompt costs about a minute of pure prefill before the
/// first token is emitted.
///
/// WHY A PREFIX AND NOT AN ARBITRARY SUFFIX. The recurrent half of the model
/// — the GatedDeltaNet convolution state and the SSM state — cannot be
/// truncated: `trim()` on the attention caches only rewinds their offset, it
/// cannot rewind the recurrent state. A cached prefix can therefore only be
/// reused when the cached token sequence is a PREFIX of the new prompt: the
/// state then continues forward instead of being rewound. A prompt that
/// diverges from the cached sequence has to start from a fresh cache.
///
/// THE TOKEN SEQUENCE IS THE CONTRACT. The cached state is only meaningful
/// together with the exact token ids it was built from. Reusing it for a
/// different sequence would feed the model a state that does not match its
/// input, so the caller records the sequence as it writes the cache and this
/// type refuses to continue a cache whose tokens do not match the prompt.
public struct Qwen36MTPPrefixCache {
    /// Tokens the cache currently represents, oldest first.
    public private(set) var tokens: [Int] = []

    /// Minimum number of cached tokens worth continuing from.
    ///
    /// A continuation only pays off when it skips real work. Below this many
    /// tokens the caller is better off with a clean prefill: the tail still
    /// has to be prefilled, and the reused state buys almost nothing.
    public let minimumReusableTokens: Int

    public init(minimumReusableTokens: Int = 512) {
        self.minimumReusableTokens = Swift.max(0, minimumReusableTokens)
    }

    /// Number of tokens the cache represents.
    public var tokenCount: Int { tokens.count }

    /// True when the cache holds no tokens.
    public var isEmpty: Bool { tokens.isEmpty }

    /// Drop the cached sequence. The caller drops the matching cache state.
    public mutating func reset() {
        tokens.removeAll(keepingCapacity: true)
    }

    /// Record the tokens the cache now represents.
    ///
    /// The caller MUST pass the exact sequence written into the cache: the
    /// seed plus every committed token. A mismatch here would let a later
    /// request continue a cache that does not match its prompt.
    public mutating func record(_ newTokens: [Int]) {
        guard !newTokens.isEmpty else { return }
        tokens.append(contentsOf: newTokens)
    }

    /// Length of the common prefix between the cached sequence and
    /// `candidate`.
    public func commonPrefixLength(with candidate: [Int]) -> Int {
        let limit = Swift.min(tokens.count, candidate.count)
        var index = 0
        while index < limit, tokens[index] == candidate[index] {
            index += 1
        }
        return index
    }

    /// True when `candidate` extends the cached sequence.
    ///
    /// "Extends" means the cached tokens are a prefix of the candidate AND the
    /// candidate is strictly longer: a prompt identical to the cached one has
    /// nothing to continue, and a shorter one would need the recurrent state
    /// rewound, which this model cannot do.
    public func canContinue(with candidate: [Int]) -> Bool {
        guard tokens.count >= minimumReusableTokens else { return false }
        guard candidate.count > tokens.count else { return false }
        return commonPrefixLength(with: candidate) == tokens.count
    }
}
