import CryptoKit
import Darwin
import Foundation
@testable import MLXFastServing
import Testing

struct MTPPinnedFilesTests {
    @Test("A FIFO without a writer is rejected without blocking startup")
    func fifoIsRejectedWithoutWriter() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("model.safetensors")
        try #require(mkfifo(path.path, 0o600) == 0)
        let finished = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            defer { finished.signal() }
            do {
                try MTPPinnedFiles.verifyFile(path, bytes: 0, sha256: String(repeating: "0", count: 64))
                Issue.record("A FIFO was accepted as pinned model data")
            } catch {}
        }
        let promptReturn = finished.wait(timeout: .now() + .milliseconds(500))
        if promptReturn == .timedOut {
            // Unblock the vulnerable implementation so the red run itself
            // remains bounded and does not leave a waiting thread behind.
            let writer = open(path.path, O_WRONLY | O_NONBLOCK | O_CLOEXEC)
            if writer >= 0 { close(writer) }
            #expect(finished.wait(timeout: .now() + .seconds(2)) == .success)
        }
        #expect(promptReturn == .success)
    }

    @Test("Content growth after the size check consumes at most one extra byte")
    func grownContentsAreBounded() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("model.safetensors")
        let grown = Data(repeating: 0x61, count: 2 << 20)
        try grown.write(to: path)
        let handle = try FileHandle(forReadingFrom: path)
        defer { try? handle.close() }
        let digest = SHA256.hash(data: grown).map { String(format: "%02x", $0) }.joined()
        // The real descriptor presents a larger body than the metadata snapshot.
        // Even its correct whole-body digest cannot authorize extra bytes.
        #expect(throws: (any Error).self) {
            try MTPPinnedFiles.verifyContents(handle, bytes: 4, sha256: digest)
        }
        #expect(try handle.offset() <= 5)
    }

    @Test("Content verification accepts exact bytes and rejects truncated data", arguments: [0, 4, (1 << 20) + 1])
    func exactAndTruncatedContents(bytes: Int) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("model.safetensors")
        let data = Data(repeating: 0x62, count: bytes)
        try data.write(to: path)
        let handle = try FileHandle(forReadingFrom: path)
        defer { try? handle.close() }
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        try MTPPinnedFiles.verifyContents(handle, bytes: UInt64(bytes), sha256: digest)
        #expect(try handle.offset() == UInt64(bytes))
        try handle.seek(toOffset: 0)
        #expect(throws: (any Error).self) {
            try MTPPinnedFiles.verifyContents(handle, bytes: UInt64(bytes + 1), sha256: digest)
        }
    }

    @Test("Pinned bytes reject mismatched digest, size, and a symbolic link")
    func invalidPinnedFiles() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = Data("pinned head fixture".utf8)
        let path = directory.appendingPathComponent("model.safetensors")
        try data.write(to: path)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        try MTPPinnedFiles.verifyFile(path, bytes: UInt64(data.count), sha256: digest)
        #expect(throws: (any Error).self) { try MTPPinnedFiles.verifyFile(path, bytes: UInt64(data.count + 1), sha256: digest) }
        #expect(throws: (any Error).self) { try MTPPinnedFiles.verifyFile(path, bytes: UInt64(data.count), sha256: String(repeating: "0", count: 64)) }
        let alias = directory.appendingPathComponent("alias.safetensors")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: path)
        #expect(throws: (any Error).self) { try MTPPinnedFiles.verifyFile(alias, bytes: UInt64(data.count), sha256: digest) }
    }

    @Test("A correctly hashed file writable by another user is refused")
    func writablePinIsRejected() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = Data("pinned".utf8)
        let path = directory.appendingPathComponent("model.safetensors")
        try data.write(to: path)
        try FileManager.default.setAttributes([.posixPermissions: 0o666], ofItemAtPath: path.path)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(throws: (any Error).self) { try MTPPinnedFiles.verifyFile(path, bytes: UInt64(data.count), sha256: digest) }
    }

    @Test("Compiled target pin matches all ten frozen published snapshot records")
    func targetManifestMatchesFrozenFixture() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let fixture = try String(contentsOf: root.appendingPathComponent("fixtures/reference_qwen3_8_27b_4bit.sha256"), encoding: .utf8)
        let expected = fixture.split(separator: "\n").filter { !$0.hasPrefix("#") }.map(String.init)
        #expect(MTPPinnedFiles.targetManifest.split(separator: "\n").map(String.init) == expected)
        #expect(expected.count == 10)
    }

    @Test("A writable snapshot directory is refused before loading")
    func unsafeDirectoryIsRejected() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        try MTPPinnedFiles.verifyDirectory(directory)
        try FileManager.default.setAttributes([.posixPermissions: 0o777], ofItemAtPath: directory.path)
        #expect(throws: (any Error).self) { try MTPPinnedFiles.verifyDirectory(directory) }
    }
}
