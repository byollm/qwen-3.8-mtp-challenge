import Foundation

// Round-13 generic A/B toggle + fire-proof. LOCAL BENCH ONLY — this file is
// never part of a submission (it lives in Sources for the toggle to be visible
// to the model module, but candidates must gate on it so the OFF path is
// byte-identical to base, and submissions strip it).
//
// Contract for each candidate implementation:
//   - When `_r13CandidateEnabled == true`, run the NEW path and do
//     `_r13FireCount += 1` once per activation (the fire-proof).
//   - When false, run the EXACT original base path (no behavior change).
// The paired-A/B harness flips this global in-process (single-load multi-leg).
public nonisolated(unsafe) var _r13CandidateEnabled: Bool = false
public nonisolated(unsafe) var _r13FireCount: Int = 0
