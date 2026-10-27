import Foundation
import Testing
@testable import MLXFastModel

// THE PREFIX-CACHE CONTRACT.
//
// The native MTP worker re-prefills the whole prompt for every HTTP request.
// On the pinned model that is the dominant cost: prefill runs at roughly
// 5 ms/token against roughly 22 ms/token for decode, so a 10.8k-token prompt
// spends about a minute in prefill before the first token is emitted.
//
// Reusing a cached prompt across requests is only possible when the cached
// token sequence is a PREFIX of the new prompt: the GatedDeltaNet state (the
// recurrent half of the model) cannot be rewound, so a diverging prompt has
// to start from a fresh cache. This file pins the decision that decides which
// of the two happens, because getting it wrong does not crash — it silently
// feeds the model a state that does not match its prompt.

@Suite
struct Qwen36MTPPrefixCacheTests {
    @Test
    func anEmptyCacheNeverContinues() {
        let cache = Qwen36MTPPrefixCache(minimumReusableTokens: 4)
        #expect(cache.isEmpty)
        #expect(cache.tokenCount == 0)
        #expect(!cache.canContinue(with: [1, 2, 3, 4, 5]))
    }

    @Test
    func aCacheBelowTheMinimumNeverContinues() {
        var cache = Qwen36MTPPrefixCache(minimumReusableTokens: 4)
        cache.record([1, 2, 3])
        #expect(cache.tokenCount == 3)
        #expect(!cache.canContinue(with: [1, 2, 3, 4]))
    }

    @Test
    func anExtendingPromptContinues() {
        var cache = Qwen36MTPPrefixCache(minimumReusableTokens: 4)
        cache.record([1, 2, 3, 4])
        #expect(cache.canContinue(with: [1, 2, 3, 4, 5, 6]))
        #expect(cache.commonPrefixLength(with: [1, 2, 3, 4, 5, 6]) == 4)
    }

    @Test
    func aDivergingPromptDoesNotContinue() {
        var cache = Qwen36MTPPrefixCache(minimumReusableTokens: 4)
        cache.record([1, 2, 3, 4])
        #expect(!cache.canContinue(with: [1, 2, 9, 4, 5]))
        #expect(cache.commonPrefixLength(with: [1, 2, 9, 4, 5]) == 2)
    }

    @Test
    func anIdenticalOrShorterPromptDoesNotContinue() {
        var cache = Qwen36MTPPrefixCache(minimumReusableTokens: 4)
        cache.record([1, 2, 3, 4])
        // Nothing to continue: the prompt would only duplicate the cached state.
        #expect(!cache.canContinue(with: [1, 2, 3, 4]))
        // Shorter: the recurrent state would have to be rewound.
        #expect(!cache.canContinue(with: [1, 2]))
    }

    @Test
    func recordingAccumulatesTheServedSequence() {
        var cache = Qwen36MTPPrefixCache(minimumReusableTokens: 1)
        cache.record([1, 2])
        cache.record([3])
        #expect(cache.tokens == [1, 2, 3])
        #expect(cache.tokenCount == 3)
        cache.reset()
        #expect(cache.isEmpty)
        #expect(cache.tokens.isEmpty)
    }

    @Test
    func theCommittedTokensArePartOfTheServedSequence() {
        // The session records the seed and then every committed token: the
        // cache represents the whole conversation, so the next request's
        // prompt — which carries the previous answer — extends it.
        var cache = Qwen36MTPPrefixCache(minimumReusableTokens: 1)
        cache.record([10, 11, 12])
        cache.record([13, 14])
        #expect(cache.canContinue(with: [10, 11, 12, 13, 14, 15]))
        #expect(!cache.canContinue(with: [10, 11, 12, 13, 99]))
    }
}
