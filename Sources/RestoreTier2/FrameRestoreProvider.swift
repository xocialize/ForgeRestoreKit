// FrameRestoreProvider.swift — RestoreTier2
//
// The narrow seam between the Restore family and whatever runs the models (BRIDGE-070).
//
// 🔑 **This file is deliberately small, and staying small is its job.** The Kit never sees model
// names, engines, or weights — it names JOBS. An adapter (ForgeCore's `EngineFrameRestoreProvider`)
// maps each job to whichever backer the measured table currently selects, and the RECEIPT carries
// which backer actually ran. That split is what lets the routing change on evidence (BRIDGE-069)
// without a single Kit-side edit — and what keeps this product importable by a host with no GPU
// toolchain, no 26.x floor, and no model downloads.
//
// ⚠️ Never a god-provider, never a generic `ModelRunning` (architecture §6.1). If a job doesn't fit
// this vocabulary, that is a design conversation, not a new protocol method with an `Any` payload.

import Foundation

/// What the caller wants done to a frame. The vocabulary is the D2 decision (2026-07-31): three
/// jobs, matching the measured capability groups — not the backer list, which is longer and churns.
public enum RestoreJob: String, Sendable, Codable, CaseIterable {
    /// Remove sensor noise. ⚠️ Whether the backer honours `strength` is a per-backer fact the
    /// receipt reports — a blind denoiser returns `appliedStrength == nil` and a UI greys the dial.
    case denoise
    /// Remove camera-shake / motion blur.
    case motionDeblur
    /// Remove out-of-focus blur.
    case defocusDeblur
}

/// One frame in, one frame out, plus the facts a receipt needs. Encoded-image bytes (PNG/JPEG) so
/// the seam stays Foundation-only; the provider owns decode/encode.
public struct FrameRestoreResult: Sendable {
    /// Encoded restored frame, same dimensions as the input.
    public let imageData: Data
    /// The strength the backer actually applied — `nil` means "this backer has no such dial",
    /// which is a fact for the UI, not an error (the BRIDGE-062 lesson: the seam must deliver what
    /// the receipt needs, never force the Kit to assume).
    public let appliedStrength: Float?
    /// The backer that actually ran (e.g. "SCUNet", "NAFNet .signage") — receipts name the engine
    /// that did the work, always.
    public let engineName: String

    public init(imageData: Data, appliedStrength: Float?, engineName: String) {
        self.imageData = imageData
        self.appliedStrength = appliedStrength
        self.engineName = engineName
    }
}

/// The seam. Implemented by an engine adapter (category B); consumed by Kits and apps (category A/C).
public protocol FrameRestoreProvider: Sendable {
    /// Restore one encoded frame.
    /// - Parameters:
    ///   - imageData: PNG or JPEG bytes.
    ///   - job: what to fix.
    ///   - strength: `0…1`, `nil` = backer default. Backers without the dial ignore it and report
    ///     `appliedStrength: nil`.
    func restore(imageData: Data, job: RestoreJob, strength: Float?) async throws -> FrameRestoreResult

    /// The backer a job would route to right now, without running it — lets a UI label the dial
    /// ("Denoise — SCUNet") and a receipt predict its engine line before work starts.
    func engineName(for job: RestoreJob) -> String
}
