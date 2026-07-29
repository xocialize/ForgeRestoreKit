//
// Deinterlace.swift — RestoreTier0
//
// N9 — deinterlacing. See `mlxengine-todo/GAP-PROGRAM.md` §N9.
//
// ═══════════════════════════════════════════════════════════════════════════════════════════════
// ⚠️ THIS DOES NOT FOLLOW §N9's PRESCRIPTION, AND THE ROW ANSWERS THE WRONG QUESTION
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
// §N9 concludes: *"Build `--disable-gpl`, link dynamically → the fast tier is free and native."* Its
// licensing work is careful and correct — yadif / bwdif / w3fdif / estdif really are LGPL 2.1+ and
// really are not GPL-gated, unlike nnedi and the rest. **But the binding constraint here was never the
// licence.** The project's standing doctrine is stricter than the LGPL question:
//
//     CLAUDE.md:183          "No FFmpeg, no vendored binaries, no libjxl/libvmaf/oxipng."
//     media-bridge:23        "No FFmpeg, ever (no link, no subprocess)."
//
// A dynamically-linked LGPL FFmpeg is a vendored binary by any reading, so *"the fast tier is free"*
// is not free here — it costs the net-clean guarantee that is the whole reason category A exists. And
// there is no Apple fallback: `VTFrameProcessor`'s verified seven configs contain **no deinterlacer**.
//
// **So this is clean-room, which the row itself endorses** — it says QTGMC *"can be reimplemented"* on
// exactly the same reasoning. What follows implements **edge-directed line averaging (ELA) with
// motion-adaptive temporal weighting**, a published technique that predates and is independent of any
// particular implementation. It is deliberately **not** a transcription of yadif, and is not claimed
// to match it.
//
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
// 🔑 **Why classical wins here, and it is not a cost argument.** Every learned deinterlacer is trained
// by dropping alternate lines from a *progressive* frame — which makes its two fields **co-temporal**.
// Real 50i/60i fields are a field-time apart, and **that temporal offset *is* the hard problem.** It is
// why DfConvEkSA scores 52.34 dB on a task nothing real scores 52 dB on. Also absent from every
// training set: field-order errors, blended and orphaned fields, 3:2 telecine, per-field chroma
// subsampling, dot crawl, head-switching, TBC error.
//
// 🔑 **And the structural advantage is `lossless`, which no learned method can offer.** Where the
// source field lines are *kept*, they are provably the original samples — not a reconstruction that
// scores well. For archival restoration that is not a nice-to-have, and it is asserted by test.
//

import Foundation

public enum Deinterlace {

    public enum DeinterlaceError: Error, Equatable {
        case sizeMismatch
        case tooSmall
    }

    /// Which field carries the earlier moment in time.
    ///
    /// ⚠️ **Getting this wrong is the classic failure and it does not look like a bug** — it looks like
    /// jerky, slightly-wrong motion, because the fields are replayed out of order. `CombDetector` is
    /// the cheap way to catch it, and field-order errors are one of the classes absent from every
    /// learned method's training set.
    public enum FieldOrder: Sendable, Equatable {
        case topFieldFirst
        case bottomFieldFirst
    }

    /// Which field's lines are the *real* samples in the frame being produced.
    public enum FieldParity: Sendable, Equatable {
        case top
        case bottom

        /// Row `y` belongs to this field.
        @inline(__always)
        func owns(row y: Int) -> Bool { (y % 2 == 0) == (self == .top) }
    }

    public enum Mode: Sendable, Equatable {
        /// Interpolate the missing lines from the current field only. Cheap, no temporal input, and
        /// the one part §N9 calls *"structurally beatable"* — a learned bobber is the honest place to
        /// spend a model, if one is ever spent here.
        case bob
        /// ELA spatial estimate blended toward the temporal (weave) estimate where the picture is
        /// static. **Weave is *exact* where nothing moved**, so the temporal path is not an
        /// approximation there — it is the original samples.
        case motionAdaptive
    }

    public struct Options: Sendable {
        public var mode: Mode
        /// 🔑 **Keep the source field lines untouched.** On by default. This is the property that makes
        /// classical structurally unbeatable for archival work: the retained lines are provably the
        /// original samples. Turning it off lets a filter "improve" real data, which for restoration is
        /// the wrong trade.
        public var lossless: Bool
        /// Half-width of the ELA direction search, in pixels. 0 collapses to a plain vertical average.
        public var edgeSearch: Int
        /// Motion below this (in [0,1] units) is treated as static, so the temporal estimate is
        /// trusted. Above roughly 4× this the spatial estimate is trusted entirely.
        public var motionThreshold: Float

        public init(mode: Mode = .motionAdaptive,
                    lossless: Bool = true,
                    edgeSearch: Int = 3,
                    motionThreshold: Float = 6.0 / 255.0) {
            self.mode = mode
            self.lossless = lossless
            self.edgeSearch = edgeSearch
            self.motionThreshold = motionThreshold
        }

        public static let `default` = Options()
    }

    // MARK: - Deinterlace

    /// Produce a progressive frame from `current`, keeping the lines belonging to `parity`.
    ///
    /// - Parameters:
    ///   - previous/next: the temporally adjacent frames, for the motion-adaptive path. Passing `nil`
    ///     for both degrades to `.bob` — stated rather than silently producing a worse result under the
    ///     same name.
    public static func frame(current: [Float],
                             previous: [Float]? = nil,
                             next: [Float]? = nil,
                             width: Int, height: Int,
                             parity: FieldParity,
                             options: Options = .default) throws -> [Float] {
        guard current.count == width * height else { throw DeinterlaceError.sizeMismatch }
        guard width >= 3, height >= 4 else { throw DeinterlaceError.tooSmall }
        if let previous, previous.count != current.count { throw DeinterlaceError.sizeMismatch }
        if let next, next.count != current.count { throw DeinterlaceError.sizeMismatch }

        let temporalAvailable = options.mode == .motionAdaptive && (previous != nil || next != nil)
        var out = current

        for y in 0..<height {
            if parity.owns(row: y) {
                // A source line. `lossless` keeps it bit-exact — see Options.lossless.
                if options.lossless { continue }
            }
            let above = max(0, y - 1), below = min(height - 1, y + 1)
            for x in 0..<width {
                let spatial = edgeDirectedEstimate(current, width: width, x: x,
                                                   above: above, below: below,
                                                   search: options.edgeSearch)
                guard temporalAvailable else {
                    out[y * width + x] = spatial
                    continue
                }

                // The temporal (weave) estimate: the same line from the adjacent frame, where it was a
                // *real* sample of the opposite field.
                let temporalSamples = [previous, next].compactMap { $0?[y * width + x] }
                let temporal = temporalSamples.reduce(0, +) / Float(temporalSamples.count)

                // Motion measured on lines we actually have in both frames — never on the line being
                // reconstructed, which is missing from `current` by definition.
                let motion = motionEstimate(current, previous: previous, next: next,
                                            width: width, x: x, above: above, below: below)
                let weight = min(1, max(0, motion / (options.motionThreshold * 4)))
                var value = temporal * (1 - weight) + spatial * weight

                // Clamp toward the vertical neighbourhood: a temporal estimate carried into a moving
                // region is the classic ghost, and the neighbours bound how far it may pull.
                //
                // 🚨 **But the clamp is gated by motion, and an unconditional one is a real bug —
                // caught by test.** Where nothing moved, the temporal sample is not an estimate at all,
                // it is *the original sample of the other field*. Clamping it to the vertical
                // neighbourhood then throws away exactly what the temporal path is for: on
                // line-alternating content the neighbours are identical, so an unconditional clamp
                // drags a perfect 0.8 back to 0.2 and the motion-adaptive mode collapses to bob on the
                // one case where it should be flawless. The clamp exists to distrust a sample that may
                // come from a *different moment* — if nothing moved, there is no different moment.
                let low = min(current[above * width + x], current[below * width + x])
                let high = max(current[above * width + x], current[below * width + x])
                let slack = (high - low) * 0.5 + options.motionThreshold
                let clamped = min(max(value, low - slack), high + slack)
                value = value * (1 - weight) + clamped * weight
                out[y * width + x] = value
            }
        }
        return out
    }

    /// Produce a full-height frame from a single field — the bob path, at field rate.
    public static func bob(field: [Float], fieldWidth: Int, fieldHeight: Int,
                           parity: FieldParity, options: Options = .default) throws -> [Float] {
        guard field.count == fieldWidth * fieldHeight else { throw DeinterlaceError.sizeMismatch }
        guard fieldWidth >= 3, fieldHeight >= 2 else { throw DeinterlaceError.tooSmall }
        let height = fieldHeight * 2
        var out = [Float](repeating: 0, count: fieldWidth * height)

        // Scatter the field onto its own rows…
        for row in 0..<fieldHeight {
            let target = parity == .top ? row * 2 : row * 2 + 1
            for x in 0..<fieldWidth {
                out[target * fieldWidth + x] = field[row * fieldWidth + x]
            }
        }
        // …then fill the rest, which is exactly the spatial problem `frame` already solves.
        return try frame(current: out, width: fieldWidth, height: height, parity: parity,
                         options: Options(mode: .bob, lossless: true,
                                          edgeSearch: options.edgeSearch,
                                          motionThreshold: options.motionThreshold))
    }

    // MARK: - Estimators

    /// Edge-directed line averaging: try several directions across the two neighbouring lines and take
    /// the one whose samples agree best.
    ///
    /// 🔑 **This is the whole reason not to average vertically.** On a near-horizontal edge a vertical
    /// average blends two different sides of the edge and produces a staircase; following the edge
    /// direction keeps it intact. The direction is chosen by local agreement, so a flat region falls
    /// back to the vertical average on its own.
    static func edgeDirectedEstimate(_ plane: [Float], width: Int, x: Int,
                                     above: Int, below: Int, search: Int) -> Float {
        let aboveRow = above * width, belowRow = below * width
        @inline(__always) func sample(_ row: Int, _ column: Int) -> Float {
            plane[row + min(max(column, 0), width - 1)]
        }

        var bestScore = Float.greatestFiniteMagnitude
        var bestValue = (sample(aboveRow, x) + sample(belowRow, x)) / 2

        for offset in -search...search {
            // A 3-wide agreement score, so a single noisy sample cannot select a direction.
            var score: Float = 0
            for tap in -1...1 {
                score += abs(sample(aboveRow, x + offset + tap) - sample(belowRow, x - offset + tap))
            }
            // Bias toward vertical: ties must not pick a slanted direction, which is how ELA produces
            // its characteristic streaks on noise.
            score += Float(abs(offset)) * 1e-3
            if score < bestScore {
                bestScore = score
                bestValue = (sample(aboveRow, x + offset) + sample(belowRow, x - offset)) / 2
            }
        }
        return bestValue
    }

    /// Motion at `x`, measured on the lines present in both frames.
    static func motionEstimate(_ current: [Float], previous: [Float]?, next: [Float]?,
                               width: Int, x: Int, above: Int, below: Int) -> Float {
        var total: Float = 0
        var count: Float = 0
        for other in [previous, next].compactMap({ $0 }) {
            total += abs(other[above * width + x] - current[above * width + x])
            total += abs(other[below * width + x] - current[below * width + x])
            count += 2
        }
        return count > 0 ? total / count : 0
    }
}

// MARK: - Comb detection

/// Is this frame interlaced, and is the field order right?
///
/// ⚠️ **Field-order errors and progressive-in-an-interlaced-container are both common and both silent.**
/// Deinterlacing progressive content softens it for nothing; using the wrong order replays motion out
/// of sequence. Neither throws an error anywhere in a pipeline, so detection has to be explicit.
public enum CombDetector {

    public struct Result: Sendable, Equatable {
        /// Mean comb energy, normalized. Higher means stronger row-alternating structure.
        public let combEnergy: Double
        /// Above the threshold — the frame carries interlacing artefacts.
        public let isInterlaced: Bool
    }

    /// Detect vertical comb structure.
    ///
    /// The measure is the classic one: for three vertically adjacent samples `a, b, c`, the product
    /// `(a − b)·(c − b)` is **large and positive only when `b` departs from both neighbours in the same
    /// direction** — which is exactly what a field of a moving object does when woven between two
    /// fields of another. A smooth gradient gives a *negative* product and a flat area gives zero, so
    /// neither is mistaken for combing.
    public static func detect(_ plane: [Float], width: Int, height: Int,
                              threshold: Double = 0.0015) -> Result {
        guard width > 0, height >= 3 else { return Result(combEnergy: 0, isInterlaced: false) }
        var total = 0.0
        var count = 0.0
        for y in 1..<(height - 1) {
            for x in 0..<width {
                let a = Double(plane[(y - 1) * width + x])
                let b = Double(plane[y * width + x])
                let c = Double(plane[(y + 1) * width + x])
                let product = (a - b) * (c - b)
                if product > 0 { total += product }
                count += 1
            }
        }
        let energy = count > 0 ? total / count : 0
        return Result(combEnergy: energy, isInterlaced: energy > threshold)
    }
}
