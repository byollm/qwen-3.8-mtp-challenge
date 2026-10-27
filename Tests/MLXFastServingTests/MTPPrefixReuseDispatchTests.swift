import Foundation
import Testing
@testable import MLXFastServing

// THE PREFIX-REUSE DISPATCH.
//
// The serving layer keeps ONE session alive across requests: that session owns
// the model's cache, and the cache is the prefix cache. The iterator is where
// the decision is made — a session that can continue its cache is started
// through the reuse path, and a session that cannot is started the ordinary
// way. Getting this wrong is silent: the request still answers, it just pays
// the full prefill again.

/// A session that records how it was started.
private class RecordingSession: MTPServingSession {
    var beganWith: [Int]?
    var reusedWith: [Int]?
    private(set) var rounds: [Int] = []
    func begin(seedTokens: [Int]) throws { beganWith = seedTokens }
    func round(depth: Int) throws -> [Int] {
        rounds.append(depth)
        return [7]
    }
}

/// A session that can continue its cache.
private final class ReusableRecordingSession: RecordingSession, MTPPrefixReusableSession {
    private(set) var servedTokens: [Int] = []
    private(set) var reuseCalls = 0
    func beginForServingReusingPrefix(seedTokens: [Int]) throws {
        reuseCalls += 1
        reusedWith = seedTokens
    }
}

@Suite
struct MTPPrefixReuseDispatchTests {
    @Test
    func anOrdinarySessionIsStartedTheOrdinaryWay() {
        let session = RecordingSession()
        var iterator = MTPTokenIterator(session: session, seedTokens: [1, 2, 3], maxTokens: 1)
        #expect(iterator.next() == 7)
        #expect(session.beganWith == [1, 2, 3])
        #expect(session.reusedWith == nil)
    }

    @Test
    func aReusableSessionIsStartedThroughTheReusePath() {
        let session = ReusableRecordingSession()
        var iterator = MTPTokenIterator(session: session, seedTokens: [1, 2, 3], maxTokens: 1)
        #expect(iterator.next() == 7)
        #expect(session.reusedWith == [1, 2, 3])
        #expect(session.beganWith == nil)
    }

    @Test
    func theSessionIsStartedExactlyOnce() {
        let session = ReusableRecordingSession()
        var iterator = MTPTokenIterator(session: session, seedTokens: [1, 2, 3], maxTokens: 3)
        while iterator.next() != nil {}
        // The prefill happens once per request, not once per token.
        #expect(session.reuseCalls == 1)
        #expect(session.rounds.count == 3)
    }
}
