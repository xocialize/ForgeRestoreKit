//
// FilmGrainParameters.swift — RestoreTier0
//
// The AFGS1 parameter set, luma half. See `AFGS1Spec.swift` for the conformance banner.
//

import Foundation

/// One knee of the piecewise-linear scaling curve: at luma `value`, grain is scaled by `scaling`.
///
/// 🔑 **The scaling curve is where a grain preset actually lives.** Amplitude as a function of
/// brightness is what separates "digital noise" from "film" — real emulsion grain peaks in the
/// midtones and falls off in the blacks and the shoulder, and a flat curve reads as video noise no
/// matter how good the AR filter is.
public struct ScalingPoint: Sendable, Equatable, Comparable {
    /// Luma value the knee sits at, 0…255.
    public var value: Int
    /// Grain scale at that luma, 0…255.
    public var scaling: Int

    public init(value: Int, scaling: Int) {
        self.value = value
        self.scaling = scaling
    }

    public static func < (a: ScalingPoint, b: ScalingPoint) -> Bool { a.value < b.value }
}

public struct FilmGrainParameters: Sendable {

    public enum ValidationError: Error, Equatable {
        /// `arCoefficients.count` must equal `2·lag·(lag+1)`.
        case coefficientCountMismatch(expected: Int, got: Int)
        case lagOutOfRange(Int)
        /// AFGS1 allows at most 14 scaling points.
        case tooManyScalingPoints(Int)
        case scalingPointsNotAscending
        case scalingPointOutOfRange(ScalingPoint)
        case gaussianSequenceWrongLength(Int)
    }

    /// Frame seed. **The whole reason this type carries a seed** — and why `CIRandomGenerator` is
    /// unusable for this job: it exposes no seed and no properties, so a grain field cannot be
    /// reproduced across renders, tiles or undo steps. **Grain that shimmers when the user pans is an
    /// immediate quality complaint**, and it is the failure a seedless generator guarantees.
    public var seed: UInt16

    /// AR filter lag, 0…3. `0` means no AR filtering at all — white grain, which is the *Gaussian*
    /// preset. Higher lag correlates neighbouring samples, which is what makes grain look like clumps
    /// of silver rather than per-pixel noise, and is also what makes it *bigger*.
    public var arCoefficientLag: Int

    /// `2·lag·(lag+1)` coefficients in scan order, each in [-128, 127] as the spec's
    /// `ar_coeffs_y_plus_128 − 128`.
    public var arCoefficients: [Int8]

    /// `ar_coeff_shift_minus_6 + 6`, i.e. 6…9. Larger means weaker AR feedback.
    public var arCoefficientShift: Int

    /// Scales the initial Gaussian draw down before filtering, 0…3. Higher means finer, weaker grain.
    public var grainScaleShift: Int

    /// `grain_scaling_minus_8 + 8`, i.e. 8…11 — the shift applied when grain meets the picture.
    public var grainScalingShift: Int

    /// The piecewise-linear scaling curve, ascending by `value`, at most 14 knees. **Empty means no
    /// luma grain at all.**
    public var scalingPoints: [ScalingPoint]

    /// Blend the 2-sample seam between neighbouring 32×32 blocks. Off produces visible 32-px tiling.
    public var overlap: Bool

    /// A global multiplier on the scaling curve, applied at synthesis. Separated from the curve itself
    /// so an "Amount" slider does not have to rewrite the preset's *shape* — the thing that makes it
    /// that preset.
    public var amount: Double

    /// 🚨 The 2048-entry Gaussian source. Defaults to the **documented substitute**, not the normative
    /// AV1 table — see `AFGS1Spec`'s conformance banner. Inject the real table for conformance.
    public var gaussianSequence: [Int16]

    public init(seed: UInt16 = 0x1234,
                arCoefficientLag: Int = 2,
                arCoefficients: [Int8] = [],
                arCoefficientShift: Int = 7,
                grainScaleShift: Int = 0,
                grainScalingShift: Int = 8,
                scalingPoints: [ScalingPoint] = [],
                overlap: Bool = true,
                amount: Double = 1.0,
                gaussianSequence: [Int16] = AFGS1.gaussianSequence) {
        self.seed = seed
        self.arCoefficientLag = arCoefficientLag
        self.arCoefficients = arCoefficients
        self.arCoefficientShift = arCoefficientShift
        self.grainScaleShift = grainScaleShift
        self.grainScalingShift = grainScalingShift
        self.scalingPoints = scalingPoints
        self.overlap = overlap
        self.amount = amount
        self.gaussianSequence = gaussianSequence
    }

    /// Validate before synthesis. **The coefficient count is checked rather than padded**: a lag/tap
    /// mismatch silently truncates the neighbourhood and produces grain that is subtly the wrong size,
    /// which is invisible in a unit test and obvious on a 40-inch panel.
    public func validate() throws {
        guard (0...3).contains(arCoefficientLag) else {
            throw ValidationError.lagOutOfRange(arCoefficientLag)
        }
        let expected = AFGS1.arCoefficientCount(lag: arCoefficientLag)
        guard arCoefficients.count == expected else {
            throw ValidationError.coefficientCountMismatch(expected: expected, got: arCoefficients.count)
        }
        guard scalingPoints.count <= 14 else {
            throw ValidationError.tooManyScalingPoints(scalingPoints.count)
        }
        for point in scalingPoints {
            guard (0...255).contains(point.value), (0...255).contains(point.scaling) else {
                throw ValidationError.scalingPointOutOfRange(point)
            }
        }
        for i in 1..<max(1, scalingPoints.count) where scalingPoints[i].value <= scalingPoints[i - 1].value {
            throw ValidationError.scalingPointsNotAscending
        }
        guard gaussianSequence.count == 2048 else {
            throw ValidationError.gaussianSequenceWrongLength(gaussianSequence.count)
        }
    }

    /// The 256-entry scaling LUT, piecewise-linear between knees and flat outside them.
    /// `amount` is folded in here so synthesis stays a pure lookup.
    public func scalingLUT() -> [Int] {
        var lut = [Int](repeating: 0, count: 256)
        guard let first = scalingPoints.first, let last = scalingPoints.last else { return lut }

        for i in 0..<min(first.value, 256) { lut[i] = first.scaling }
        for i in 0..<max(0, scalingPoints.count - 1) {
            let a = scalingPoints[i], b = scalingPoints[i + 1]
            let deltaX = b.value - a.value
            let deltaY = b.scaling - a.scaling
            guard deltaX > 0 else { continue }
            // Fixed-point slope, matching the spec's 16-bit reciprocal form.
            let delta = deltaY * ((65536 + (deltaX >> 1)) / deltaX)
            for x in 0..<deltaX where a.value + x < 256 {
                lut[a.value + x] = a.scaling + ((x * delta + 32768) >> 16)
            }
        }
        for i in min(last.value, 256)..<256 { lut[i] = last.scaling }

        guard amount != 1.0 else { return lut }
        return lut.map { AFGS1.clip(Int((Double($0) * amount).rounded()), 0, 255) }
    }
}
