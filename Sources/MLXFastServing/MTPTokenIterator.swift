import Foundation
import MLXLMCommon

/// The model boundary. A session owns all mutable prefill/decode caches.
public protocol MTPServingSession: AnyObject {
    func begin(seedTokens: [Int]) throws
    func round(depth: Int) throws -> [Int]
    var statistics: MTPServingStatistics? { get }
}

public extension MTPServingSession {
    var statistics: MTPServingStatistics? { nil }
}

/// A session whose cache can be continued by a prompt that EXTENDS the token
/// sequence the cache already represents.
///
/// The capability is SEPARATE from `MTPServingSession` on purpose: a session
/// that cannot be reused still serves requests, and a caller must not assume
/// that any session's cache survives its request.
public protocol MTPPrefixReusableSession: MTPServingSession {
    /// Tokens the session's cache currently represents.
    var servedTokens: [Int] { get }
    /// Begin a request, reusing the previous request's cache when the prompt
    /// extends the cached sequence.
    func beginForServingReusingPrefix(seedTokens: [Int]) throws
}

public final class MTPGenerationFailure: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: (any Error)?
    public init() {}
    public var error: (any Error)? { lock.withLock { stored } }
    func record(_ error: any Error) { lock.withLock { stored = error } }
}

public struct MTPTokenIterator: TokenIteratorProtocol {
    public let maxTokens: Int?
    public private(set) var tokenCount = 0
    public private(set) var promptPrefillTime: TimeInterval = 0
    private let session: any MTPServingSession
    private let seedTokens: [Int]
    private let maxDepth: Int
    private let failure: MTPGenerationFailure
    private var began = false
    private var finished = false
    private var queue: [Int] = []

    public init(session: any MTPServingSession, seedTokens: [Int], maxTokens: Int,
                maxDepth: Int = 8, failure: MTPGenerationFailure = .init()) {
        self.session = session
        self.seedTokens = seedTokens
        self.maxTokens = maxTokens
        self.maxDepth = maxDepth
        self.failure = failure
    }

    public mutating func next() -> Int? {
        guard !finished, tokenCount < (maxTokens ?? .max) else { return nil }
        do {
            try Task.checkCancellation()
            if !began {
                if let reusable = session as? any MTPPrefixReusableSession {
                    try reusable.beginForServingReusingPrefix(seedTokens: seedTokens)
                } else {
                    try session.begin(seedTokens: seedTokens)
                }
                // Prefill happens inside next(), so the common loop already
                // includes it in promptTime. Adding it here would double count.
                began = true
            }
            try Task.checkCancellation()
            if queue.isEmpty {
                let remaining = (maxTokens ?? .max) - tokenCount
                queue = try session.round(depth: Swift.min(maxDepth, Swift.max(0, remaining - 1)))
            }
            try Task.checkCancellation()
            guard !queue.isEmpty else { finished = true; return nil }
            tokenCount += 1
            return queue.removeFirst()
        } catch {
            finished = true
            failure.record(error)
            return nil
        }
    }
}
