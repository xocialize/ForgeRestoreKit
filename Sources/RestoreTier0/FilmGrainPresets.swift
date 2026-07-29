//
// FilmGrainPresets.swift — RestoreTier0
//
// The preset mapping from `GAP-PROGRAM.md` §N2. **⚠️ The competitor's own grain documentation 404'd,
// so the mapping from their preset *names* to AFGS1 parameters is inferred, not reverse-engineered.**
// The parameter meanings are not in doubt; which of their presets a given curve corresponds to is.
//
// 🔑 **Size and Amount are separate axes and must stay separate.** Size is the AR lag plus coefficient
// magnitude — it changes the *correlation structure*, i.e. how big a clump of grain is. Amount is a
// gain on the scaling curve — it changes how strongly that structure shows. A UI that folds them into
// one "strength" slider loses the control that makes grain look like film rather than noise.
//

import Foundation

public enum FilmGrainPreset: String, Sendable, CaseIterable {
    /// `ar_coeff_lag = 0`, all AR coefficients zero — uncorrelated grain. The honest "digital noise"
    /// preset, and the physical one: the scaling curve here comes from the photon-noise model rather
    /// than from taste.
    case gaussian
    /// Luma-only grain (`num_cb_points = num_cr_points = 0`), moderate correlation.
    case grey
    /// `lag = 3` (24 taps), low-pass coefficients, and a curve **peaked in the midtones and rolled off
    /// in blacks and whites** — the tonal signature of emulsion.
    case silverRich
}

public extension FilmGrainParameters {

    /// Build a preset. `size` (0…1) scales the correlation structure, `amount` (0…1) the visibility.
    ///
    /// ⚠️ Note what is *not* here: a "film stock" name. Matching a specific stock is the residual-fit
    /// estimator's job (below), not a preset's — a preset is a shape, a fit is a measurement.
    static func preset(_ preset: FilmGrainPreset,
                       seed: UInt16 = 0x1234,
                       size: Double = 0.5,
                       amount: Double = 0.5) -> FilmGrainParameters {
        let size = min(max(size, 0), 1)
        let amount = min(max(amount, 0), 1)

        switch preset {
        case .gaussian:
            return FilmGrainParameters(
                seed: seed,
                arCoefficientLag: 0,
                arCoefficients: [],
                arCoefficientShift: 6,
                grainScaleShift: 0,
                grainScalingShift: 8,
                scalingPoints: photonNoiseCurve(),
                overlap: true,
                amount: amount)

        case .grey:
            let lag = size < 0.5 ? 1 : 2
            return FilmGrainParameters(
                seed: seed,
                arCoefficientLag: lag,
                arCoefficients: lowPassCoefficients(lag: lag, strength: size),
                arCoefficientShift: 7,
                grainScaleShift: 0,
                grainScalingShift: 8,
                scalingPoints: neutralCurve(),
                overlap: true,
                amount: amount)

        case .silverRich:
            return FilmGrainParameters(
                seed: seed,
                arCoefficientLag: 3,
                arCoefficients: lowPassCoefficients(lag: 3, strength: 0.5 + size * 0.5),
                arCoefficientShift: 7,
                grainScaleShift: 0,
                grainScalingShift: 8,
                scalingPoints: midtonePeakedCurve(),
                overlap: true,
                amount: amount)
        }
    }

    /// A low-pass causal neighbourhood whose weights fall off with distance from the centre.
    ///
    /// 🔑 **Two parameters, and conflating them is the mistake this function was rewritten to fix.**
    /// The first version normalized a *fixed* total gain across however many taps the lag provided, so
    /// raising the lag spread the same energy thinner and the nearest-neighbour weight fell from
    /// 24/128 at lag 1 to 4/128 at lag 3. Measured result: correlation went **0.33 → 0.17 → 0.096** as
    /// lag rose 1 → 2 → 3. Grain got *finer* as the "size" control went up — the exact inverse of the
    /// intent, and invisible to every test except one that measures correlation directly.
    ///
    /// 🔑 **And the second measurement corrected the fix.** Widening the falloff with lag did not help
    /// either — because for an *auto-regressive* process, correlation **length is governed by total
    /// feedback gain, not by kernel width**. The filter is IIR: a strong nearest-neighbour coefficient
    /// propagates recursively and produces long correlation, while the same gain smeared across a wide
    /// neighbourhood weakens the recursion. Measured, at fixed gain, distance-3 correlation was *lower*
    /// at lag 3 (0.23) than at lag 1 (0.28), despite lag 1 having no tap that reaches 3 px at all.
    ///
    /// So the final shape is: **falloff is fixed**, total gain is the size control, and `lag` supplies
    /// the 2-D isotropy and smoothness of the clump rather than its size.
    ///
    /// ⚠️ **This contradicts the plan's preset table**, which maps *Size* to *"`ar_coeff_lag` +
    /// coefficient magnitude"*. Magnitude dominates; lag on its own can make grain finer. Worth
    /// carrying back if the AFGS1 preset mapping is ever revisited.
    static func lowPassCoefficients(lag: Int, strength: Double) -> [Int8] {
        let taps = AFGS1.arTaps(lag: lag)
        guard !taps.isEmpty else { return [] }
        let strength = min(max(strength, 0), 1)

        // Falloff is FIXED, not a function of lag or strength — see the note above the function.
        let raw: [Double] = taps.map { tap in
            let distance = Double(tap.deltaRow * tap.deltaRow + tap.deltaColumn * tap.deltaColumn)
                .squareRoot()
            return Foundation.exp(-distance / 0.9)
        }
        let total = raw.reduce(0, +)
        // Total feedback gain — the size control. Held below 1.0 in filter terms (128 in the spec's
        // fixed point) so the recursion stays stable.
        let budget = 128.0 * (0.50 + 0.42 * strength)
        return raw.map { Int8(AFGS1.clip(Int(($0 / total * budget).rounded()), -128, 127)) }
    }

    /// Photon-noise shape: grain tracks √signal, so amplitude *rises* with brightness. This is the one
    /// curve here derived from physics rather than taste — shot noise is Poisson, so its standard
    /// deviation goes as the square root of the electron count.
    static func photonNoiseCurve() -> [ScalingPoint] {
        stride(from: 0, through: 255, by: 36).map { value in
            let normalized = Double(value) / 255
            return ScalingPoint(value: value,
                                scaling: AFGS1.clip(Int((normalized.squareRoot() * 200).rounded()), 0, 255))
        }
    }

    /// Broadly flat with a gentle shoulder roll-off — the general-purpose curve.
    static func neutralCurve() -> [ScalingPoint] {
        [ScalingPoint(value: 0, scaling: 40),
         ScalingPoint(value: 32, scaling: 110),
         ScalingPoint(value: 96, scaling: 150),
         ScalingPoint(value: 160, scaling: 150),
         ScalingPoint(value: 216, scaling: 110),
         ScalingPoint(value: 255, scaling: 60)]
    }

    /// Peaked in the midtones, rolled off hard in blacks and whites — emulsion's signature. The blacks
    /// matter most perceptually: grain in the shadows is what reads as "dirty" rather than "filmic".
    static func midtonePeakedCurve() -> [ScalingPoint] {
        [ScalingPoint(value: 0, scaling: 8),
         ScalingPoint(value: 24, scaling: 48),
         ScalingPoint(value: 64, scaling: 140),
         ScalingPoint(value: 112, scaling: 190),
         ScalingPoint(value: 160, scaling: 175),
         ScalingPoint(value: 208, scaling: 96),
         ScalingPoint(value: 255, scaling: 24)]
    }
}
