// The seam's contract, pinned. RestoreTier2 is protocols-only, so what a test can protect is the
// shape: the job vocabulary is the D2 decision and public API from the first tag — additions are
// fine, renames are breaking changes someone must notice.
import Foundation
import Testing

@testable import RestoreTier2

@Suite("FrameRestoreProvider seam")
struct FrameRestoreProviderTests {

    @Test("the D2 job vocabulary is exactly the decided three")
    func jobVocabulary() {
        #expect(Set(RestoreJob.allCases.map(\.rawValue)) == ["denoise", "motionDeblur", "defocusDeblur"])
    }

    @Test("jobs round-trip through Codable by stable raw value")
    func jobsRoundTrip() throws {
        for job in RestoreJob.allCases {
            let data = try JSONEncoder().encode(job)
            #expect(try JSONDecoder().decode(RestoreJob.self, from: data) == job)
        }
    }

    /// A mock proving the protocol is implementable without MLX, an engine, or AppKit — the
    /// net-clean guarantee exercised, not asserted.
    struct Mock: FrameRestoreProvider {
        func restore(imageData: Data, job: RestoreJob, strength: Float?) async throws -> FrameRestoreResult {
            FrameRestoreResult(imageData: imageData,
                               appliedStrength: job == .denoise ? strength : nil,
                               engineName: "mock")
        }
        func engineName(for job: RestoreJob) -> String { "mock" }
    }

    @Test("result carries the receipt fields")
    func receiptFields() async throws {
        let r = try await Mock().restore(imageData: Data([1, 2, 3]), job: .denoise, strength: 0.5)
        #expect(r.appliedStrength == 0.5)
        #expect(r.engineName == "mock")
        let r2 = try await Mock().restore(imageData: Data(), job: .motionDeblur, strength: 0.5)
        #expect(r2.appliedStrength == nil, "a dial-less job must report nil, not echo the ask")
    }
}
