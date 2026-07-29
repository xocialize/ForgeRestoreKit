//
// FilmGrainTemplate.swift — RestoreTier0
//
// The 82×73 luma grain template: a Gaussian draw, then a causal AR filter over it.
//
// 🔑 **The "AR is sequential so it must be slow" objection dissolves on inspection, and this is why
// the whole stage stays on the CPU.** 70×76 filtered positions × ≤24 taps ≈ **144k multiply-adds,
// once per image** — tens of microseconds in scalar Swift. libplacebo, dav1d and FFmpeg all generate
// on the CPU for the same reason. Do not parallelize it; the template is not the hot path, the
// per-pixel application is.
//

import Foundation

/// A generated grain template. Values are in `[GrainMin, GrainMax]` (8-bit: −128…127).
public struct FilmGrainTemplate: Sendable, Equatable {
    public let samples: [Int]
    public let width: Int
    public let height: Int

    public init(samples: [Int], width: Int, height: Int) {
        self.samples = samples; self.width = width; self.height = height
    }

    @inline(__always)
    public subscript(x: Int, y: Int) -> Int { samples[y * width + x] }
}

public extension FilmGrainTemplate {

    /// Generate the luma template for `parameters`.
    ///
    /// Two stages, and the ordering is load-bearing: the whole array is drawn from the Gaussian source
    /// first, *then* filtered in place, so each AR output feeds the next — the filter is genuinely
    /// recursive, not a convolution over the unfiltered field. Filtering into a copy would produce a
    /// visibly different (and much weaker) correlation structure.
    static func luma(parameters: FilmGrainParameters) -> FilmGrainTemplate {
        let width = AFGS1.lumaTemplateWidth
        let height = AFGS1.lumaTemplateHeight
        var samples = [Int](repeating: 0, count: width * height)

        // Stage 1 — the Gaussian draw. `grainScaleShift` thins it before any filtering.
        let drawShift = 4 + parameters.grainScaleShift
        var register = AFGS1.RandomRegister(state: parameters.seed)
        let hasGrain = !parameters.scalingPoints.isEmpty
        for i in 0..<samples.count {
            // The register is walked even when there is no grain, so that a zero-grain frame does not
            // silently desynchronize a caller that later enables grain on the same seed.
            let draw = register.next(bits: 11)
            samples[i] = hasGrain ? AFGS1.round2(Int(parameters.gaussianSequence[draw]), drawShift) : 0
        }

        // Stage 2 — the causal AR filter, in place.
        let taps = AFGS1.arTaps(lag: parameters.arCoefficientLag)
        if !taps.isEmpty && hasGrain {
            let lag = parameters.arCoefficientLag
            let shift = parameters.arCoefficientShift
            for y in lag..<height {
                for x in lag..<(width - lag) {
                    var accumulator = 0
                    for (index, tap) in taps.enumerated() {
                        let sample = samples[(y + tap.deltaRow) * width + (x + tap.deltaColumn)]
                        accumulator += sample * Int(parameters.arCoefficients[index])
                    }
                    let position = y * width + x
                    samples[position] = AFGS1.clip(samples[position] + AFGS1.round2(accumulator, shift),
                                                   AFGS1.grainMin8Bit, AFGS1.grainMax8Bit)
                }
            }
        }

        return FilmGrainTemplate(samples: samples, width: width, height: height)
    }
}
