import XCTest
@testable import RestoreTier0

/// "Match the original grain" — fitting a grain model to a denoise residual.
///
/// 🔑 **The tests assert a PERCEPTUAL round trip, not coefficient recovery.** Fitting an AR model to a
/// single noisy realization is ill-posed — many parameter sets explain the same field — so demanding
/// the original coefficients back would be measuring the fixture, not the feature. What must hold is
/// that synthesizing from the estimate produces noise of the same *strength* and the same *coarseness*.
final class FilmGrainEstimatorTests: XCTestCase {

    // MARK: - Fixtures

    /// A smooth picture — the "denoised" image. Deliberately low-gradient so most blocks are flat.
    private func smoothImage(_ width: Int, _ height: Int) -> [Float] {
        (0..<(width * height)).map { i in
            let x = Float(i % width) / Float(width), y = Float(i / width) / Float(height)
            return 0.25 + 0.5 * (0.5 + 0.5 * Foundation.sin(x * 2) * Foundation.cos(y * 1.5))
        }
    }

    private func addNoise(_ base: [Float], sigma: Float, seed: UInt32) -> [Float] {
        var state = seed &* 2_654_435_761 &+ 1
        func uniform() -> Float {
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float((state >> 8) & 0xffff) / 65535 - 0.5
        }
        // Sum of three uniforms ≈ Gaussian, and deterministic.
        return base.map { $0 + (uniform() + uniform() + uniform()) * sigma * 2.0 }
    }

    /// Synthesize real AFGS1 grain onto a base — the honest input, since it is what we claim to match.
    private func addSynthesizedGrain(_ base: [Float], width: Int, height: Int,
                                     parameters: FilmGrainParameters) throws -> [Float] {
        try FilmGrainSynthesizer(parameters: parameters).apply(to: base, width: width, height: height)
    }

    private func sigma(_ a: [Float], _ b: [Float]) -> Double {
        let d = zip(a, b).map { Double($0 - $1) }
        let mean = d.reduce(0, +) / Double(d.count)
        return (d.reduce(0.0) { $0 + pow($1 - mean, 2) } / Double(d.count)).squareRoot()
    }

    private func correlation(_ a: [Float], _ b: [Float], width: Int, height: Int) -> Double {
        var product = 0.0, energy = 0.0
        for y in 2..<(height - 2) {
            for x in 2..<(width - 3) {
                let i = y * width + x
                let r0 = Double(a[i] - b[i]), r1 = Double(a[i + 1] - b[i + 1])
                product += r0 * r1; energy += r0 * r0
            }
        }
        return energy > 0 ? product / energy : 0
    }

    // MARK: - Stage 1 · flat blocks

    /// The stage that decides whether the model learns noise or content. A frame that is smooth on the
    /// left and busy on the right must select from the left.
    func testFlatBlocksAvoidStructure() {
        let width = 256, height = 128
        var image = [Float](repeating: 0.5, count: width * height)
        for y in 0..<height {
            for x in 128..<width {
                image[y * width + x] = (x + y) % 4 < 2 ? 0.2 : 0.8   // hard texture
            }
        }
        let blocks = FilmGrainEstimator.flatBlocks(in: image, width: width, height: height,
                                                   options: .default)
        XCTAssertFalse(blocks.isEmpty)
        XCTAssertTrue(blocks.allSatisfy { $0.x < 128 },
                      "every fitted block must come from the smooth half, got \(blocks.map(\.x))")
    }

    /// An image with no flat area at all must fail loudly. A confident wrong answer here gets
    /// synthesized over the picture as "grain".
    func testAllStructureFailsHonestly() {
        let width = 128, height = 128
        let busy = (0..<(width * height)).map { i -> Float in
            (i % 3 == 0) ? 0.1 : ((i % 3 == 1) ? 0.9 : 0.4)
        }
        var options = FilmGrainEstimator.Options.default
        options.minimumFlatBlocks = 1000                    // force the failure path
        XCTAssertThrowsError(try FilmGrainEstimator.fit(original: busy, denoised: busy,
                                                        width: width, height: height,
                                                        options: options)) { error in
            guard case FilmGrainEstimator.EstimationError.notEnoughFlatBlocks = error else {
                return XCTFail("wrong error: \(error)")
            }
        }
    }

    // MARK: - Stage 2 · the solver

    func testCholeskySolvesAKnownSystem() {
        // [[4,1],[1,3]] x = [1,2]  ->  x = [1/11, 7/11]
        let solved = FilmGrainEstimator.solveSymmetric([4, 1, 1, 3], [1, 2], 2)
        let x = try! XCTUnwrap(solved)
        XCTAssertEqual(x[0], 1.0 / 11, accuracy: 1e-9)
        XCTAssertEqual(x[1], 7.0 / 11, accuracy: 1e-9)
    }

    func testCholeskyRejectsANonPositiveDefiniteMatrix() {
        XCTAssertNil(FilmGrainEstimator.solveSymmetric([0, 0, 0, 0], [1, 1], 2))
    }

    /// ⚠️ **The stability cap, which is not optional.** A fit on a noisy realization can return a total
    /// gain at or above unity; the synthesizer's recursion would then run away and clip the template to
    /// a flat ±127 field — grain whose histogram still looks plausible.
    func testQuantizationCapsRunawayGain() {
        let runaway = [Double](repeating: 0.5, count: 12)   // total gain 6.0
        let (coefficients, shift) = FilmGrainEstimator.quantize(runaway)
        let total = coefficients.reduce(0.0) { $0 + abs(Double($1)) } / Double(1 << shift)
        XCTAssertLessThan(total, 1.0, "total feedback \(total) would make the recursion diverge")
        XCTAssertTrue(coefficients.allSatisfy { $0 != 0 }, "the shape must survive the rescale")
    }

    // MARK: - The round trip

    /// 🔑 **The headline claim.** Fit a model to real synthesized grain, re-synthesize from the
    /// estimate, and the noise strength must come back close.
    func testRoundTripReproducesNoiseStrength() throws {
        let width = 192, height = 160
        let clean = smoothImage(width, height)
        let truth = FilmGrainParameters.preset(.grey, seed: 0x1234, size: 0.5, amount: 0.55)
        let noisy = try addSynthesizedGrain(clean, width: width, height: height, parameters: truth)

        let estimate = try FilmGrainEstimator.fit(original: noisy, denoised: clean,
                                                  width: width, height: height)
        let reproduced = try addSynthesizedGrain(clean, width: width, height: height,
                                                 parameters: estimate.parameters)

        let sourceSigma = sigma(noisy, clean)
        let matchedSigma = sigma(reproduced, clean)
        // Measured 0.92× on this fixture (0.0406 → 0.0374); 20% is real headroom over that while
        // still being tight enough to catch a regression. An earlier 40% would have passed anything.
        XCTAssertGreaterThan(sourceSigma, 0.002, "the fixture must carry real grain")
        XCTAssertEqual(matchedSigma, sourceSigma, accuracy: sourceSigma * 0.2,
                       "σ \(sourceSigma) → \(matchedSigma)")
    }

    /// …and the *coarseness* must come back too. Strength alone is satisfiable by white noise of the
    /// right amplitude, which would look nothing like the source.
    func testRoundTripReproducesCoarseness() throws {
        let width = 192, height = 160
        let clean = smoothImage(width, height)
        let coarse = FilmGrainParameters.preset(.silverRich, seed: 0x99, size: 1.0, amount: 0.6)
        let noisy = try addSynthesizedGrain(clean, width: width, height: height, parameters: coarse)

        let estimate = try FilmGrainEstimator.fit(original: noisy, denoised: clean,
                                                  width: width, height: height)
        let reproduced = try addSynthesizedGrain(clean, width: width, height: height,
                                                 parameters: estimate.parameters)

        let sourceCorrelation = correlation(noisy, clean, width: width, height: height)
        let matchedCorrelation = correlation(reproduced, clean, width: width, height: height)
        // Measured 1.10× on this fixture (0.426 → 0.467). Bounded on both sides: fitting *coarser*
        // than the source is as wrong as fitting finer, and only a two-sided bound catches a
        // runaway AR gain that would otherwise look like a very good match.
        XCTAssertGreaterThan(sourceCorrelation, 0.15, "the fixture must be genuinely coarse")
        XCTAssertGreaterThan(matchedCorrelation, sourceCorrelation * 0.75,
                             "too fine: \(sourceCorrelation) → \(matchedCorrelation)")
        XCTAssertLessThan(matchedCorrelation, sourceCorrelation * 1.4,
                          "too coarse: \(sourceCorrelation) → \(matchedCorrelation)")
    }

    /// Fine grain must not come back coarse. Together with the test above this pins that the AR fit
    /// carries information rather than returning a constant.
    func testFineAndCoarseSourcesAreDistinguished() throws {
        let width = 192, height = 160
        let clean = smoothImage(width, height)

        func fittedCorrelation(_ parameters: FilmGrainParameters) throws -> Double {
            let noisy = try addSynthesizedGrain(clean, width: width, height: height, parameters: parameters)
            let estimate = try FilmGrainEstimator.fit(original: noisy, denoised: clean,
                                                      width: width, height: height)
            let reproduced = try addSynthesizedGrain(clean, width: width, height: height,
                                                     parameters: estimate.parameters)
            return correlation(reproduced, clean, width: width, height: height)
        }

        // Measured: gaussian fits to r ≈ −0.04, silverRich to r ≈ 0.47 — a wide, unambiguous gap.
        let fine = try fittedCorrelation(.preset(.gaussian, seed: 0x11, amount: 0.6))
        let coarse = try fittedCorrelation(.preset(.silverRich, seed: 0x11, size: 1.0, amount: 0.6))
        XCTAssertLessThan(abs(fine), 0.15, "a white source must fit ~white, got \(fine)")
        XCTAssertGreaterThan(coarse, fine + 0.3,
                             "a coarse source must fit coarser than a fine one: fine=\(fine) coarse=\(coarse)")
    }

    /// The residual is what gets modelled, so a heavier source must fit a stronger curve.
    func testStrongerNoiseFitsAStrongerCurve() throws {
        let width = 160, height = 128
        let clean = smoothImage(width, height)

        func fittedSigma(_ amplitude: Float) throws -> Float {
            let noisy = addNoise(clean, sigma: amplitude, seed: 0x2024)
            return try FilmGrainEstimator.fit(original: noisy, denoised: clean,
                                              width: width, height: height).residualSigma
        }
        let light = try fittedSigma(0.004), heavy = try fittedSigma(0.020)
        XCTAssertGreaterThan(heavy, light * 3, "light=\(light) heavy=\(heavy)")
    }

    /// A noiseless input must not invent grain — the "already clean" case.
    func testNoResidualMeansNoGrain() throws {
        let width = 128, height = 128
        let clean = smoothImage(width, height)
        let estimate = try FilmGrainEstimator.fit(original: clean, denoised: clean,
                                                  width: width, height: height)
        XCTAssertEqual(estimate.residualSigma, 0, accuracy: 1e-6)
        XCTAssertTrue(estimate.parameters.scalingPoints.allSatisfy { $0.scaling == 0 },
                      "a zero residual must produce a zero curve, not a floor")
    }

    // MARK: - Report

    /// `flatBlockFraction` is the honest input to a "match source" confidence indicator: a busy frame
    /// yields a weaker fit and the UI should be able to say so.
    func testFlatFractionFallsOnBusierContent() throws {
        let width = 192, height = 160
        let clean = smoothImage(width, height)
        let noisy = addNoise(clean, sigma: 0.01, seed: 7)

        var busy = clean
        for y in 0..<height {
            for x in (width / 2)..<width where (x + y) % 5 < 2 { busy[y * width + x] += 0.3 }
        }
        let busyNoisy = addNoise(busy, sigma: 0.01, seed: 7)

        var options = FilmGrainEstimator.Options.default
        options.flatBlockFraction = 1.0        // ask for everything, so the *scores* decide
        let smoothFit = try FilmGrainEstimator.fit(original: noisy, denoised: clean,
                                                   width: width, height: height, options: options)
        XCTAssertEqual(smoothFit.flatBlockFraction, 1.0, accuracy: 0.01)

        let quarter = FilmGrainEstimator.Options.default
        let busyFit = try FilmGrainEstimator.fit(original: busyNoisy, denoised: busy,
                                                 width: width, height: height, options: quarter)
        XCTAssertLessThan(busyFit.flatBlockFraction, 0.5)
        XCTAssertGreaterThan(busyFit.flatBlockCount, 0)
    }

    func testSizeMismatchIsRejected() {
        XCTAssertThrowsError(try FilmGrainEstimator.fit(
            original: [Float](repeating: 0, count: 10),
            denoised: [Float](repeating: 0, count: 20), width: 10, height: 1)) {
            XCTAssertEqual($0 as? FilmGrainEstimator.EstimationError, .sizeMismatch)
        }
    }
}

