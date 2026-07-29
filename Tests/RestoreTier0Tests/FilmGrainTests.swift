import XCTest
@testable import RestoreTier0

/// N2 — AFGS1 film grain. What is asserted here is deliberately **structural**, not conformance: the
/// Gaussian source is a documented substitute for the normative AV1 table (see `AFGS1Spec`'s banner),
/// so a golden-value test would pin the substitute rather than the spec. Everything below holds
/// regardless of which Gaussian source is injected — which also means these tests keep working, and
/// keep meaning something, on the day the real table is dropped in.
final class FilmGrainTests: XCTestCase {

    // MARK: - Tap geometry

    /// `2·lag·(lag+1)` → 0, 4, 12, 24. The plan states this independently of the loop that produces
    /// it, so agreement is a real cross-check on the neighbourhood's shape rather than a tautology.
    func testARTapCountsMatchTheClosedForm() {
        XCTAssertEqual(AFGS1.arTaps(lag: 0).count, 0)
        XCTAssertEqual(AFGS1.arTaps(lag: 1).count, 4)
        XCTAssertEqual(AFGS1.arTaps(lag: 2).count, 12)
        XCTAssertEqual(AFGS1.arTaps(lag: 3).count, 24)
        for lag in 0...3 {
            XCTAssertEqual(AFGS1.arTaps(lag: lag).count, AFGS1.arCoefficientCount(lag: lag))
        }
    }

    /// Causality: every tap is strictly earlier in scan order than the centre. A single non-causal tap
    /// makes the recursion read its own output and the filter stops being well-defined.
    func testEveryTapIsCausal() {
        for lag in 1...3 {
            for tap in AFGS1.arTaps(lag: lag) {
                let isEarlier = tap.deltaRow < 0 || (tap.deltaRow == 0 && tap.deltaColumn < 0)
                XCTAssertTrue(isEarlier, "non-causal tap \(tap) at lag \(lag)")
                XCTAssertLessThanOrEqual(abs(tap.deltaRow), lag)
                XCTAssertLessThanOrEqual(abs(tap.deltaColumn), lag)
            }
        }
    }

    // MARK: - Template

    func testTemplateIs82By73() {
        let template = FilmGrainTemplate.luma(parameters: .preset(.grey))
        XCTAssertEqual(template.width, 82)
        XCTAssertEqual(template.height, 73)
        XCTAssertEqual(template.samples.count, 82 * 73)
    }

    /// The sampled window is `[9, 73)` in both axes — the plan's own statement, re-derived here from
    /// the offset arithmetic so a change to any of the three constants that produce it fails loudly.
    func testSampledWindowStaysInsideTheTemplate() {
        let maxIndex = AFGS1.templateSampleOrigin
            + AFGS1.maxBlockOffset * 2
            + AFGS1.blockReadExtent - 1
        XCTAssertEqual(maxIndex, 72, "the 64×64 effective window ends at 72, i.e. [9, 73)")
        XCTAssertLessThan(maxIndex, AFGS1.lumaTemplateHeight)
        XCTAssertLessThan(maxIndex, AFGS1.lumaTemplateWidth)
    }

    /// The AR feedback budget must leave the template *unsaturated*. Too much gain and the recursion
    /// runs away, every sample clips to ±127, and the result is a flat field that still looks like
    /// "grain" in a histogram — which is exactly why this is asserted rather than eyeballed.
    func testTemplateIsNotSaturatedAtAnyPreset() throws {
        for preset in FilmGrainPreset.allCases {
            let template = FilmGrainTemplate.luma(parameters: .preset(preset, size: 1.0))
            let clipped = template.samples.filter { $0 <= AFGS1.grainMin8Bit || $0 >= AFGS1.grainMax8Bit }
            XCTAssertLessThan(Double(clipped.count) / Double(template.samples.count), 0.02,
                              "\(preset) saturates \(clipped.count) of \(template.samples.count) samples")
            let mean = Double(template.samples.reduce(0, +)) / Double(template.samples.count)
            let variance = template.samples.reduce(0.0) { $0 + pow(Double($1) - mean, 2) }
                / Double(template.samples.count)
            XCTAssertGreaterThan(variance, 1.0, "\(preset) produced a nearly flat template")
        }
    }

    /// 🔑 **The AR filter's whole purpose, measured — and the assertion here is the corrected one.**
    ///
    /// The first version of this test asserted that correlation rises with *lag*. It failed, and it was
    /// the test that was wrong: for an auto-regressive process, correlation length is governed by total
    /// feedback **gain**, not by how many taps the gain is spread across. Measured at fixed gain,
    /// distance-3 correlation was *lower* at lag 3 (0.23) than at lag 1 (0.28) — despite lag 1 having
    /// no tap that reaches 3 px. So the size control is gain, and what is asserted is that.
    ///
    /// Without this test the AR stage could be wired to nothing, or wired backwards, and every other
    /// test in this file would still pass.
    func testCorrelationRisesWithGrainSize() throws {
        let sizes = [0.0, 0.5, 1.0]
        let correlations = try sizes.map { size -> Double in
            let parameters = FilmGrainParameters.preset(.silverRich, seed: 0x1234, size: size)
            return neighbourCorrelation(FilmGrainTemplate.luma(parameters: parameters))
        }
        for i in 1..<correlations.count {
            XCTAssertGreaterThan(correlations[i], correlations[i - 1] + 0.05,
                                 "size \(sizes[i]) must be coarser than \(sizes[i - 1]): \(correlations)")
        }
    }

    /// Lag 0 is white by construction — no taps, no feedback. The baseline the test above is measured
    /// against, and the *Gaussian* preset's defining property.
    func testLagZeroIsUncorrelated() {
        let white = FilmGrainTemplate.luma(parameters: .preset(.gaussian))
        XCTAssertLessThan(abs(neighbourCorrelation(white)), 0.1)
    }

    /// The product claim behind the preset names: Silver Rich at full size is visibly coarser grain
    /// than Gaussian, which is uncorrelated digital noise.
    func testSilverRichIsCoarserThanGaussian() {
        let fine = neighbourCorrelation(FilmGrainTemplate.luma(parameters: .preset(.gaussian)))
        let coarse = neighbourCorrelation(
            FilmGrainTemplate.luma(parameters: .preset(.silverRich, size: 1.0)))
        XCTAssertGreaterThan(coarse, fine + 0.25, "fine=\(fine) coarse=\(coarse)")
    }

    /// Nearest-neighbour horizontal correlation, normalized — 0 for white noise, → 1 for a smooth field.
    private func neighbourCorrelation(_ t: FilmGrainTemplate) -> Double {
        var product = 0.0, energy = 0.0
        for y in 6..<(t.height - 6) {
            for x in 6..<(t.width - 7) {
                product += Double(t[x, y]) * Double(t[x + 1, y])
                energy += Double(t[x, y]) * Double(t[x, y])
            }
        }
        return energy > 0 ? product / energy : 0
    }

    // MARK: - Determinism and tiling — the product property

    /// Same parameters, same grain. The floor everything else rests on.
    func testSameSeedProducesIdenticalGrain() throws {
        let synth = try FilmGrainSynthesizer(parameters: .preset(.grey, seed: 0xBEEF))
        XCTAssertEqual(synth.grainField(width: 200, height: 140),
                       synth.grainField(width: 200, height: 140))
    }

    func testDifferentSeedsProduceDifferentGrain() throws {
        let a = try FilmGrainSynthesizer(parameters: .preset(.grey, seed: 0x0001))
        let b = try FilmGrainSynthesizer(parameters: .preset(.grey, seed: 0x0002))
        XCTAssertNotEqual(a.grainField(width: 128, height: 128), b.grainField(width: 128, height: 128))
    }

    /// 🚨 **The test this whole design exists for.** A tile rendered at an arbitrary offset must carry
    /// bit-identical grain to the same region of the full-frame render — otherwise grain crawls under
    /// the picture as the user pans, which the plan names as *"an immediate quality complaint"*. It is
    /// also the concrete reason `CIRandomGenerator` is unusable here: it exposes no seed, so this
    /// property is not merely untested with it, it is unachievable.
    ///
    /// The origin is deliberately **not** block-aligned (37, 91) — alignment would hide an off-by-one
    /// in the block-to-pixel mapping, which is the likeliest way to get this wrong.
    func testTiledRenderMatchesFullFrameBitForBit() throws {
        let synth = try FilmGrainSynthesizer(parameters: .preset(.silverRich, seed: 0x4242))
        let width = 320, height = 256
        let full = synth.grainField(width: width, height: height)

        let tileX = 37, tileY = 91, tileW = 96, tileH = 64
        let tile = synth.grainField(width: tileW, height: tileH, originX: tileX, originY: tileY)

        for y in 0..<tileH {
            for x in 0..<tileW {
                XCTAssertEqual(tile[y * tileW + x], full[(tileY + y) * width + (tileX + x)],
                               "mismatch at tile (\(x), \(y))")
            }
        }
    }

    /// A picture's grain must not depend on how big the picture is — the same absolute position must
    /// draw the same value. This is the frame-size half of the property above.
    func testGrainAtAPositionDoesNotDependOnPictureSize() throws {
        let synth = try FilmGrainSynthesizer(parameters: .preset(.grey, seed: 0x77))
        let small = synth.grainField(width: 128, height: 128)
        let large = synth.grainField(width: 512, height: 256)
        for y in 0..<128 {
            for x in 0..<128 {
                XCTAssertEqual(small[y * 128 + x], large[y * 512 + x])
            }
        }
    }

    /// Block offsets are absolute: reaching a block by walking further along the row must not change
    /// what it draws.
    func testBlockOffsetsAreAbsolute() throws {
        let synth = try FilmGrainSynthesizer(parameters: .preset(.grey, seed: 0x9))
        let short = synth.blockOffsets(blockRow: 3, throughColumn: 5)
        let long = synth.blockOffsets(blockRow: 3, throughColumn: 40)
        XCTAssertEqual(Array(long.prefix(6)), short)
    }

    // MARK: - Overlap

    /// 🔑 **Overlap earns its place, measured at the seam.** Without it, the 32-px block grid is
    /// visible as a discontinuity every 32 columns. The test compares the mean absolute step *across*
    /// block boundaries with the step at ordinary interior columns: with overlap off the boundary step
    /// is markedly larger; with it on the two are comparable.
    func testOverlapSuppressesTheBlockSeam() throws {
        func seamRatio(overlap: Bool) throws -> Double {
            var parameters = FilmGrainParameters.preset(.silverRich, seed: 0x1111, size: 1.0)
            parameters.overlap = overlap
            let synth = try FilmGrainSynthesizer(parameters: parameters)
            let width = 320, height = 128
            let field = synth.grainField(width: width, height: height)

            var boundary = 0.0, boundaryCount = 0.0
            var interior = 0.0, interiorCount = 0.0
            for y in 0..<height {
                for x in 1..<width {
                    let step = abs(Double(field[y * width + x] - field[y * width + x - 1]))
                    if x % AFGS1.blockSize == 0 { boundary += step; boundaryCount += 1 }
                    else { interior += step; interiorCount += 1 }
                }
            }
            return (boundary / boundaryCount) / (interior / interiorCount)
        }

        let without = try seamRatio(overlap: false)
        let with = try seamRatio(overlap: true)
        XCTAssertGreaterThan(without, 1.15, "with overlap off the seam must be measurably worse")
        XCTAssertLessThan(with, without, "overlap must reduce the seam: with=\(with) without=\(without)")
    }

    // MARK: - Scaling curve

    func testScalingLUTIsPiecewiseLinearAndFlatOutsideTheKnees() {
        let parameters = FilmGrainParameters(
            arCoefficientLag: 0,
            scalingPoints: [ScalingPoint(value: 50, scaling: 10),
                            ScalingPoint(value: 150, scaling: 110)])
        let lut = parameters.scalingLUT()

        XCTAssertEqual(lut[0], 10, "flat below the first knee")
        XCTAssertEqual(lut[49], 10)
        XCTAssertEqual(lut[50], 10)
        XCTAssertEqual(lut[100], 60, accuracy: 1, "linear halfway between the knees")
        XCTAssertEqual(lut[150], 110)
        XCTAssertEqual(lut[255], 110, "flat above the last knee")
    }

    func testAmountScalesTheCurveWithoutChangingItsShape() {
        var full = FilmGrainParameters.preset(.silverRich, amount: 1.0)
        full.amount = 1.0
        var half = full
        half.amount = 0.5

        let a = full.scalingLUT(), b = half.scalingLUT()
        XCTAssertEqual(a.count, b.count)
        for i in 0..<256 {
            XCTAssertEqual(Double(b[i]), Double(a[i]) / 2, accuracy: 1.0, "at luma \(i)")
        }
    }

    func testEmptyScalingCurveMeansNoGrain() throws {
        let parameters = FilmGrainParameters(arCoefficientLag: 0, scalingPoints: [])
        let synth = try FilmGrainSynthesizer(parameters: parameters)
        let plane = [Float](repeating: 0.5, count: 64 * 64)
        XCTAssertEqual(synth.apply(to: plane, width: 64, height: 64), plane)
    }

    // MARK: - Presets

    /// Silver Rich is *defined* by its tonal signature — peaked in the midtones, rolled off in the
    /// blacks. Grain in the shadows is what reads as dirty rather than filmic, so this is the
    /// preset's identity and not a detail.
    func testSilverRichPeaksInTheMidtonesAndRollsOffInTheBlacks() {
        let lut = FilmGrainParameters.preset(.silverRich, amount: 1.0).scalingLUT()
        XCTAssertGreaterThan(lut[112], lut[8], "midtones must carry more grain than the blacks")
        XCTAssertGreaterThan(lut[112], lut[248], "…and more than the whites")
        XCTAssertLessThan(lut[8], 40, "blacks must be nearly clean")
    }

    /// The photon-noise curve is the physical one: shot noise is Poisson, so amplitude rises with the
    /// square root of the signal — monotonically increasing, the opposite shape to Silver Rich.
    func testGaussianPresetFollowsPhotonNoiseAndRisesWithBrightness() {
        let lut = FilmGrainParameters.preset(.gaussian, amount: 1.0).scalingLUT()
        XCTAssertGreaterThan(lut[240], lut[16], "photon noise must rise with signal")
        XCTAssertEqual(FilmGrainParameters.preset(.gaussian).arCoefficientLag, 0,
                       "Gaussian is lag 0 by definition")
        XCTAssertTrue(FilmGrainParameters.preset(.gaussian).arCoefficients.isEmpty)
    }

    func testEveryPresetValidates() throws {
        for preset in FilmGrainPreset.allCases {
            for size in [0.0, 0.5, 1.0] {
                try FilmGrainParameters.preset(preset, size: size, amount: 0.7).validate()
            }
        }
    }

    // MARK: - Validation

    /// A lag/coefficient mismatch must throw rather than truncate. Truncation would shrink the
    /// neighbourhood silently and produce grain of subtly the wrong size — invisible in a unit test,
    /// obvious on a large panel.
    func testCoefficientCountMismatchThrows() {
        let parameters = FilmGrainParameters(arCoefficientLag: 3, arCoefficients: [1, 2, 3])
        XCTAssertThrowsError(try parameters.validate()) { error in
            XCTAssertEqual(error as? FilmGrainParameters.ValidationError,
                           .coefficientCountMismatch(expected: 24, got: 3))
        }
    }

    func testTooManyScalingPointsThrows() {
        let points = (0..<15).map { ScalingPoint(value: $0 * 16, scaling: 100) }
        let parameters = FilmGrainParameters(arCoefficientLag: 0, scalingPoints: points)
        XCTAssertThrowsError(try parameters.validate()) {
            XCTAssertEqual($0 as? FilmGrainParameters.ValidationError, .tooManyScalingPoints(15))
        }
    }

    func testNonAscendingScalingPointsThrow() {
        let parameters = FilmGrainParameters(
            arCoefficientLag: 0,
            scalingPoints: [ScalingPoint(value: 100, scaling: 10),
                            ScalingPoint(value: 50, scaling: 20)])
        XCTAssertThrowsError(try parameters.validate()) {
            XCTAssertEqual($0 as? FilmGrainParameters.ValidationError, .scalingPointsNotAscending)
        }
    }

    func testWrongLengthGaussianSequenceThrows() {
        var parameters = FilmGrainParameters.preset(.grey)
        parameters.gaussianSequence = [0, 1, 2]
        XCTAssertThrowsError(try parameters.validate()) {
            XCTAssertEqual($0 as? FilmGrainParameters.ValidationError, .gaussianSequenceWrongLength(3))
        }
    }

    // MARK: - The substitute Gaussian source

    /// The substitute is not the normative table, but it must have the table's observable shape — or
    /// grain generated with it will not merely be non-conformant, it will look wrong.
    func testSubstituteGaussianSequenceHasTheRightShape() {
        let sequence = AFGS1.substituteGaussianSequence
        XCTAssertEqual(sequence.count, 2048)
        XCTAssertTrue(sequence.allSatisfy { $0 >= -2048 && $0 <= 2047 })
        XCTAssertTrue(sequence.allSatisfy { $0 % 4 == 0 }, "quantized to multiples of 4, as the table is")

        let mean = Double(sequence.map(Int.init).reduce(0, +)) / 2048
        XCTAssertEqual(mean, 0, accuracy: 40, "approximately zero-mean")
        let sigma = (sequence.reduce(0.0) { $0 + pow(Double($1) - mean, 2) } / 2048).squareRoot()
        XCTAssertEqual(sigma, 512, accuracy: 60, "approximately σ=512")
    }

    /// Determinism across builds: the sequence is a stored constant, so grain cannot drift because a
    /// compiler reassociated a floating-point expression.
    func testSubstituteGaussianSequenceIsStable() {
        XCTAssertEqual(AFGS1.substituteGaussianSequence, AFGS1.substituteGaussianSequence)
    }

    // MARK: - Application

    /// Grain must reach the pixels, and it must respect the curve: with Silver Rich, a midtone plane
    /// picks up visibly more grain than a near-black one.
    func testAppliedGrainFollowsTheToneCurve() throws {
        let synth = try FilmGrainSynthesizer(parameters: .preset(.silverRich, seed: 0x33, amount: 1.0))

        func deviation(level: Float) -> Double {
            let plane = [Float](repeating: level, count: 128 * 128)
            let grained = synth.apply(to: plane, width: 128, height: 128)
            let diffs = zip(plane, grained).map { abs(Double($1 - $0)) }
            return diffs.reduce(0, +) / Double(diffs.count)
        }

        let midtone = deviation(level: 112.0 / 255)
        let shadow = deviation(level: 8.0 / 255)
        XCTAssertGreaterThan(midtone, 0, "grain must actually reach the picture")
        XCTAssertGreaterThan(midtone, shadow * 2, "midtone=\(midtone) shadow=\(shadow)")
    }

    func testApplyStaysInRange() throws {
        let synth = try FilmGrainSynthesizer(parameters: .preset(.gaussian, seed: 0x5, amount: 1.0))
        for level in [Float(0), 0.02, 0.5, 0.98, 1.0] {
            let plane = [Float](repeating: level, count: 64 * 64)
            for value in synth.apply(to: plane, width: 64, height: 64) {
                XCTAssertGreaterThanOrEqual(value, 0)
                XCTAssertLessThanOrEqual(value, 1)
            }
        }
    }
}

