//
// BadPixelCorrection.swift — RestoreTier0
//
// N3 — hot and dead pixel correction. See `mlxengine-todo/GAP-PROGRAM.md` §N3.
//
// **A RAW feature Apple does not expose at all**, and 100% classical in every reference implementation
// surveyed. Adobe standardized the *result* as DNG opcodes 4 and 5 (`FixBadPixelsConstant` /
// `FixBadPixelsList`, TIFF tag 51008) — but ⚠️ **LibRaw will not do it for you**: it parses 51008 only
// far enough to set a presence flag. So a pipeline that wants this has to implement it.
//
// **Operates on the CFA mosaic, before demosaic**, which is the only place it is well defined: a hot
// photosite is one *sensel*, and after demosaic its error has been smeared across a neighbourhood and
// into two other colour channels. Everything below therefore works on same-colour neighbours, which on
// a Bayer grid are at ±2 — the "5×5 same-colour window" of the DNG SDK's `FixIsolatedPixel`.
//
// ═══════════════════════════════════════════════════════════════════════════════════════════════
// Two false-positive classes, and only one of them is solvable here
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
//   • 🔴 **Stars are geometrically identical to hot pixels.** A single bright sensel surrounded by
//     dark sky is *exactly* what the detector is built to find, and no amount of threshold tuning
//     separates them, because there is nothing to separate — from one frame they are the same signal.
//     **This is not a tuning problem and must not be presented as one.** The real answers are a
//     measured bad-pixel *list* (a dark frame, or DNG opcode 5 — `knownBadPixels` below), or
//     multi-frame agreement. `detectStarLikeIsolatedPoints` exists as a test so nobody later assumes
//     the detector got smarter.
//   • 🟡 **PDAF sensels are row-correlated** — regularly spaced, repeating down a frame. That
//     structure *is* detectable, so a suspicious pattern is **flagged rather than silently repaired**
//     (`phaseDetectPatternSuspected`). Correcting PDAF sites with a neighbour average is not
//     catastrophic, but it is the wrong tool and hides a masking problem that belongs upstream.
//
// ⚠️ **A recorded discrepancy that does NOT apply here.** §N3 notes that darktable's manual says it
// replaces a bad pixel with the *average* of its neighbours while its code uses the *max*, and advises
// implementing max. That is guidance for reimplementing darktable's algorithm. This implements the
// DNG SDK's `FixIsolatedPixel` instead — the row's own "algorithm to copy" — which is gradient-directed
// and strictly better on edges, so the max/mean question does not arise. Noted so the next reader does
// not "fix" this to match a note about a different algorithm.
//

import Foundation

public enum BadPixelCorrection {

    public enum CorrectionError: Error, Equatable {
        case tooSmall
        case sizeMismatch
        /// The detector flagged an implausible share of the sensor, **scattered**. Aborting beats
        /// correcting: at that density the operation stops being repair and becomes a smoothing filter
        /// applied to every isolated detail in the frame. The actionable response is to raise the
        /// threshold — the detector is mis-tuned for this content.
        case implausibleDefectDensity(fraction: Double, limit: Double)
        /// The flagged sites are dense **and regular** — a phase-detect grid, not defects.
        ///
        /// 🔑 **This is a separate case on purpose, and it was a real defect in the first version.**
        /// PDAF sites are legitimately dense (percent-scale on some sensors), so the density guard
        /// fired first and reported `implausibleDefectDensity` — which reads as "your threshold is too
        /// low" and invites exactly the wrong fix. The two conditions together are *diagnostic*: this
        /// is the wrong tool, and the answer is a PDAF mask upstream, not a tuning change.
        case phaseDetectPatternDetected(fraction: Double)
    }

    public struct Options: Sendable {
        /// How far a sensel must deviate from its same-colour median, **in multiples of the local
        /// high-frequency energy**, to be called defective.
        ///
        /// 🔑 **The normalization is the whole detector.** An absolute threshold flags every specular
        /// highlight and every fine texture; dividing by the local high-frequency energy makes the
        /// test *"is this pixel anomalous relative to how busy its neighbourhood already is"*, which
        /// automatically backs off in texture. This is RawTherapee's approach and it is why its
        /// detector is the one §N3 names.
        public var threshold: Double
        /// Absolute floor under the local energy, so a perfectly flat patch cannot make the
        /// denominator vanish and turn sensor read noise into a defect.
        public var energyFloor: Double
        public var correctHot: Bool
        public var correctDead: Bool
        /// Abort above this defect share. A real sensor's stuck-pixel population is measured in parts
        /// per million; a percent is a mis-tuned threshold.
        public var maximumDefectFraction: Double
        /// Column spacing regularity above which a pattern is called PDAF-like rather than defective.
        public var phaseDetectSuspicionRatio: Double

        public init(threshold: Double = 8.0,
                    energyFloor: Double = 1.0 / 512.0,
                    correctHot: Bool = true,
                    correctDead: Bool = true,
                    maximumDefectFraction: Double = 0.002,
                    phaseDetectSuspicionRatio: Double = 0.5) {
            self.threshold = threshold
            self.energyFloor = energyFloor
            self.correctHot = correctHot
            self.correctDead = correctDead
            self.maximumDefectFraction = maximumDefectFraction
            self.phaseDetectSuspicionRatio = phaseDetectSuspicionRatio
        }

        public static let `default` = Options()
    }

    public struct Defect: Sendable, Equatable {
        public let x: Int
        public let y: Int
        /// Signed deviation from the same-colour median, normalized by local energy. Positive = hot.
        public let score: Double
        public var isHot: Bool { score > 0 }
    }

    public struct Report: Sendable, Equatable {
        public let detected: Int
        public let corrected: Int
        public let defectFraction: Double
        /// 🟡 The detected sites look like a regular row/column pattern rather than scattered defects.
        /// **Correction still ran** — this flags that the tool may be the wrong one, not that it failed.
        public let phaseDetectPatternSuspected: Bool
    }

    // MARK: - Detect

    /// Find defective sensels in a CFA mosaic.
    ///
    /// - Parameters:
    ///   - cfa: the single-channel mosaic, normalized to [0, 1].
    ///   - knownBadPixels: sites from a measured map (a dark frame, or DNG opcode 5). These are
    ///     corrected **unconditionally** and never scored — which is the only reliable way to fix a
    ///     defect that sits on a star, and the reason this parameter exists.
    public static func detect(cfa: [Float], width: Int, height: Int,
                              knownBadPixels: [(x: Int, y: Int)] = [],
                              options: Options = .default) throws -> [Defect] {
        guard cfa.count == width * height else { throw CorrectionError.sizeMismatch }
        guard width >= 5, height >= 5 else { throw CorrectionError.tooSmall }

        var found: [Defect] = []
        var known = Set<Int>()
        for site in knownBadPixels where site.x >= 0 && site.x < width && site.y >= 0 && site.y < height {
            known.insert(site.y * width + site.x)
            found.append(Defect(x: site.x, y: site.y, score: .infinity))
        }

        // Same-colour neighbours sit at ±2 on a Bayer grid — hence the 5×5 window for a 3×3 sample set.
        for y in 2..<(height - 2) {
            for x in 2..<(width - 2) {
                let index = y * width + x
                if known.contains(index) { continue }

                var neighbours = [Float]()
                neighbours.reserveCapacity(8)
                for dy in stride(from: -2, through: 2, by: 2) {
                    for dx in stride(from: -2, through: 2, by: 2) where !(dx == 0 && dy == 0) {
                        neighbours.append(cfa[(y + dy) * width + x + dx])
                    }
                }
                neighbours.sort()
                let median = Double(neighbours[3] + neighbours[4]) / 2

                // Local high-frequency energy: mean absolute deviation of the same-colour neighbours
                // about their own median. Busy neighbourhoods raise the bar for calling a defect.
                var energy = 0.0
                for value in neighbours { energy += abs(Double(value) - median) }
                energy = max(energy / Double(neighbours.count), options.energyFloor)

                let deviation = Double(cfa[index]) - median
                let score = deviation / energy
                guard abs(score) >= options.threshold else { continue }
                if score > 0 && !options.correctHot { continue }
                if score < 0 && !options.correctDead { continue }
                found.append(Defect(x: x, y: y, score: score))
            }
        }
        return found
    }

    // MARK: - Correct

    /// Repair one sensel using the DNG SDK's `FixIsolatedPixel` rule.
    ///
    /// Four directional estimates — horizontal, vertical, and the two diagonals — each from the two
    /// same-colour neighbours along that axis, each carrying a gradient score. **Every direction whose
    /// gradient is within 1.5× of the best is averaged**, rather than taking the single best.
    ///
    /// 🔑 **Why gradient-directed and not a plain neighbourhood average.** A defect on an edge averaged
    /// in all directions pulls values across the edge and leaves a visible smudge exactly where the eye
    /// is most sensitive. Choosing by gradient follows the edge instead. And averaging *all* directions
    /// within 1.5× rather than only the minimum is what keeps a flat area from picking one arbitrary
    /// axis and inheriting its noise.
    static func repairedValue(cfa: [Float], width: Int, height: Int, x: Int, y: Int) -> Float {
        @inline(__always) func at(_ dx: Int, _ dy: Int) -> Double {
            let sx = min(max(x + dx, 0), width - 1)
            let sy = min(max(y + dy, 0), height - 1)
            return Double(cfa[sy * width + sx])
        }
        // (estimate, gradient) per axis, over same-colour neighbours at ±2.
        let axes: [(estimate: Double, gradient: Double)] = [
            ((at(-2, 0) + at(2, 0)) / 2, abs(at(-2, 0) - at(2, 0))),
            ((at(0, -2) + at(0, 2)) / 2, abs(at(0, -2) - at(0, 2))),
            ((at(-2, -2) + at(2, 2)) / 2, abs(at(-2, -2) - at(2, 2))),
            ((at(2, -2) + at(-2, 2)) / 2, abs(at(2, -2) - at(-2, 2))),
        ]
        let best = axes.map(\.gradient).min() ?? 0
        let limit = best * 1.5
        let chosen = axes.filter { $0.gradient <= limit || $0.gradient == best }
        let sum = chosen.reduce(0.0) { $0 + $1.estimate }
        return Float(sum / Double(max(1, chosen.count)))
    }

    /// Detect and repair in one pass.
    public static func detectAndCorrect(cfa: [Float], width: Int, height: Int,
                                        knownBadPixels: [(x: Int, y: Int)] = [],
                                        options: Options = .default) throws -> (cfa: [Float], report: Report) {
        let defects = try detect(cfa: cfa, width: width, height: height,
                                 knownBadPixels: knownBadPixels, options: options)
        let fraction = Double(defects.count) / Double(width * height)
        let phaseDetect = looksLikePhaseDetect(defects, height: height, options: options)
        if fraction > options.maximumDefectFraction {
            // Diagnose the pattern BEFORE blaming the threshold — see `phaseDetectPatternDetected`.
            throw phaseDetect
                ? CorrectionError.phaseDetectPatternDetected(fraction: fraction)
                : CorrectionError.implausibleDefectDensity(fraction: fraction,
                                                           limit: options.maximumDefectFraction)
        }

        // ⚠️ Read every replacement from the ORIGINAL. Writing in place lets one repaired sensel feed
        // the next, so a cluster of adjacent defects would propagate the first estimate outward
        // instead of each being solved from real data.
        var out = cfa
        for defect in defects {
            out[defect.y * width + defect.x] = repairedValue(cfa: cfa, width: width, height: height,
                                                             x: defect.x, y: defect.y)
        }

        return (out, Report(detected: defects.count,
                            corrected: defects.count,
                            defectFraction: fraction,
                            phaseDetectPatternSuspected: phaseDetect))
    }

    // MARK: - PDAF suspicion

    /// PDAF sensels repeat on a regular grid; genuine defects scatter. A pattern, not a proof, which
    /// is why the result is a flag rather than a veto.
    ///
    /// 🔑 **Two conditions, and the first one was missing from the initial version.** Clustering alone
    /// — "most detections share a row with at least two others" — is *trivially satisfied by a
    /// saturated detector*: with a mis-tuned threshold flagging 87% of the sensor, every row qualifies
    /// and pure noise gets diagnosed as a phase-detect grid. Caught by the density test, which started
    /// reporting the wrong error.
    ///
    /// The distinguishing property of PDAF is **periodicity**: the sites live on a lattice, so the
    /// overwhelming majority of rows contain *none*. Requiring low row occupancy first separates a
    /// periodic pattern from a detector that is simply firing everywhere.
    static func looksLikePhaseDetect(_ defects: [Defect], height: Int, options: Options) -> Bool {
        guard defects.count >= 12 else { return false }   // too few to call a pattern
        var perRow: [Int: Int] = [:]
        for defect in defects { perRow[defect.y, default: 0] += 1 }

        // Periodicity: a lattice leaves most rows empty. A saturated detector does not.
        let occupancy = Double(perRow.count) / Double(max(1, height))
        guard occupancy <= 0.5 else { return false }

        let clustered = perRow.values.filter { $0 >= 3 }.reduce(0, +)
        return Double(clustered) / Double(defects.count) >= options.phaseDetectSuspicionRatio
    }
}
