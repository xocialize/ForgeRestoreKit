import XCTest
@testable import RestoreTier0

/// N3 — hot/dead sensel repair on a CFA mosaic. The tests are built around the two things that make
/// this hard: not firing on texture, and being honest about the false-positive class that cannot be
/// solved from a single frame.
final class BadPixelCorrectionTests: XCTestCase {

    // MARK: - Fixtures

    /// A Bayer-like mosaic where the four colour sites sit at clearly different levels. Any correction
    /// that reaches a *different* colour's neighbours produces an obvious error, which is what makes
    /// this fixture worth more than a flat grey one.
    private func mosaic(_ width: Int, _ height: Int) -> [Float] {
        (0..<(width * height)).map { i in
            let x = i % width, y = i / width
            switch (x % 2, y % 2) {
            case (0, 0): return 0.62   // R
            case (1, 0): return 0.38   // G
            case (0, 1): return 0.36   // G
            default:     return 0.20   // B
            }
        }
    }

    private func withDefect(_ plane: [Float], _ x: Int, _ y: Int, _ value: Float,
                            width: Int) -> [Float] {
        var out = plane
        out[y * width + x] = value
        return out
    }

    // MARK: - Detection

    func testHotAndDeadPixelsAreFound() throws {
        let width = 64, height = 64
        var plane = mosaic(width, height)
        plane = withDefect(plane, 20, 20, 1.0, width: width)     // hot
        plane = withDefect(plane, 41, 33, 0.0, width: width)     // dead

        let defects = try BadPixelCorrection.detect(cfa: plane, width: width, height: height)
        XCTAssertEqual(defects.count, 2, "found \(defects.map { ($0.x, $0.y) })")
        XCTAssertTrue(defects.contains { $0.x == 20 && $0.y == 20 && $0.isHot })
        XCTAssertTrue(defects.contains { $0.x == 41 && $0.y == 33 && !$0.isHot })
    }

    /// 🔑 **The property the normalization exists for.** Fine texture must not read as a field of
    /// defects — an absolute threshold would flag most of this fixture.
    func testFineTextureIsNotFlagged() throws {
        let width = 96, height = 96
        var state: UInt32 = 0xBEEF
        let textured = (0..<(width * height)).map { i -> Float in
            state = state &* 1_664_525 &+ 1_013_904_223
            let base: Float = (i % 2 == 0) ? 0.55 : 0.30
            return base + (Float((state >> 16) & 0xffff) / 65535 - 0.5) * 0.20
        }
        let defects = try BadPixelCorrection.detect(cfa: textured, width: width, height: height)
        let fraction = Double(defects.count) / Double(width * height)
        XCTAssertLessThan(fraction, 0.001, "flagged \(defects.count) of \(width * height) textured sensels")
    }

    func testHotOnlyAndDeadOnlyModes() throws {
        let width = 48, height = 48
        var plane = mosaic(width, height)
        plane = withDefect(plane, 10, 10, 1.0, width: width)
        plane = withDefect(plane, 20, 20, 0.0, width: width)

        var hotOnly = BadPixelCorrection.Options.default
        hotOnly.correctDead = false
        let hot = try BadPixelCorrection.detect(cfa: plane, width: width, height: height, options: hotOnly)
        XCTAssertEqual(hot.count, 1)
        XCTAssertTrue(hot[0].isHot)

        var deadOnly = BadPixelCorrection.Options.default
        deadOnly.correctHot = false
        let dead = try BadPixelCorrection.detect(cfa: plane, width: width, height: height, options: deadOnly)
        XCTAssertEqual(dead.count, 1)
        XCTAssertFalse(dead[0].isHot)
    }

    // MARK: - Correction

    /// The repair must come from **same-colour** neighbours. On this fixture a naive 3×3 average would
    /// mix R with G and land far from the correct 0.62.
    func testRepairUsesSameColourNeighbours() throws {
        let width = 64, height = 64
        let clean = mosaic(width, height)
        let broken = withDefect(clean, 20, 20, 1.0, width: width)

        let (fixed, report) = try BadPixelCorrection.detectAndCorrect(
            cfa: broken, width: width, height: height)

        XCTAssertEqual(report.corrected, 1)
        XCTAssertEqual(fixed[20 * width + 20], 0.62, accuracy: 0.01,
                       "repaired to \(fixed[20 * width + 20]); a mixed-colour average would land near 0.4")
        // Nothing else moved.
        for i in 0..<clean.count where i != 20 * width + 20 {
            XCTAssertEqual(fixed[i], clean[i], "sensel \(i) changed but was not defective")
        }
    }

    /// 🔑 **Gradient direction is the reason this beats a neighbourhood average.** With a vertical edge
    /// through the defect, the repair must follow the edge (take the vertical neighbours) rather than
    /// average across it and land halfway between the two sides.
    func testRepairFollowsAnEdgeRatherThanAveragingAcrossIt() throws {
        let width = 64, height = 64
        var plane = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                // Same colour everywhere (constant parity) so only the edge matters here.
                plane[y * width + x] = x < 32 ? 0.20 : 0.80
            }
        }
        let defectX = 34, defectY = 30        // just inside the bright side
        plane[defectY * width + defectX] = 0.0

        let repaired = BadPixelCorrection.repairedValue(cfa: plane, width: width, height: height,
                                                        x: defectX, y: defectY)
        XCTAssertEqual(repaired, 0.80, accuracy: 0.01,
                       "repaired to \(repaired); averaging across the edge would give ~0.5")
    }

    /// Adjacent defects must each be solved from real data. Repairing in place would let the first
    /// estimate feed the second and propagate outward.
    func testAdjacentDefectsDoNotFeedEachOther() throws {
        let width = 64, height = 64
        let clean = mosaic(width, height)
        var broken = clean
        broken[30 * width + 30] = 1.0
        broken[30 * width + 32] = 1.0        // same colour, adjacent in the same-colour lattice

        let (fixed, report) = try BadPixelCorrection.detectAndCorrect(
            cfa: broken, width: width, height: height)
        XCTAssertEqual(report.corrected, 2)
        XCTAssertEqual(fixed[30 * width + 30], 0.62, accuracy: 0.02)
        XCTAssertEqual(fixed[30 * width + 32], 0.62, accuracy: 0.02)
    }

    // MARK: - The honest limits

    /// 🔴 **Pinned deliberately: a star is indistinguishable from a hot pixel.** This test asserts the
    /// detector *does* fire on an isolated bright point in dark surroundings, because that is the
    /// truth — the two signals are identical from one frame. Anyone who later "fixes" this by tuning
    /// the threshold will only move the boundary, not solve it. The solutions are a measured bad-pixel
    /// list or multi-frame agreement, and the API takes the former.
    func testAStarIsIndistinguishableFromAHotPixel() throws {
        let width = 64, height = 64
        var sky = [Float](repeating: 0.02, count: width * height)
        sky[30 * width + 30] = 0.95           // a star

        let defects = try BadPixelCorrection.detect(cfa: sky, width: width, height: height)
        XCTAssertTrue(defects.contains { $0.x == 30 && $0.y == 30 },
                      "the detector must be honest that it cannot tell these apart")
    }

    /// …and the escape hatch works: a measured list is corrected unconditionally and scored never, so a
    /// real defect sitting on a star still gets fixed.
    func testAKnownBadPixelIsCorrectedWithoutScoring() throws {
        let width = 48, height = 48
        let clean = mosaic(width, height)
        var broken = clean
        // A defect genuinely below the threshold. The fixture is noiseless, so the local energy floors
        // at 1/512 and even a small deviation scores highly — 0.03 was *not* mild here and tripped the
        // detector, which is why this is 0.005 (score ≈ 2.6 against a threshold of 8).
        broken[24 * width + 24] = clean[24 * width + 24] + 0.005

        let unaided = try BadPixelCorrection.detect(cfa: broken, width: width, height: height)
        XCTAssertFalse(unaided.contains { $0.x == 24 && $0.y == 24 },
                       "the fixture must be below the detection threshold for this test to mean anything")

        let (fixed, report) = try BadPixelCorrection.detectAndCorrect(
            cfa: broken, width: width, height: height, knownBadPixels: [(x: 24, y: 24)])
        XCTAssertEqual(report.corrected, 1)
        XCTAssertEqual(fixed[24 * width + 24], clean[24 * width + 24], accuracy: 0.01)
    }

    /// 🟡 PDAF sites repeat on a regular grid. That is detectable, so it is flagged rather than
    /// silently repaired — correcting them is the wrong tool and hides a masking problem upstream.
    func testRegularRowPatternIsFlaggedAsPhaseDetect() throws {
        let width = 128, height = 128
        var plane = mosaic(width, height)
        for row in stride(from: 16, to: 112, by: 24) {
            for column in stride(from: 8, to: 120, by: 24) {
                plane[row * width + column] = 1.0
            }
        }
        let (_, report) = try BadPixelCorrection.detectAndCorrect(
            cfa: plane, width: width, height: height)
        XCTAssertGreaterThan(report.detected, 12)
        XCTAssertLessThanOrEqual(report.defectFraction, 0.002, "this fixture must stay under the abort limit")
        XCTAssertTrue(report.phaseDetectPatternSuspected,
                      "a regular row/column grid must be flagged, got \(report.detected) scattered")
    }

    /// 🔑 **A DENSE regular grid must be diagnosed as PDAF, not blamed on the threshold.** PDAF sites
    /// are legitimately percent-scale on some sensors, so the density guard fires — and the first
    /// version reported `implausibleDefectDensity`, which reads as "lower your threshold" and invites
    /// exactly the wrong fix. The two conditions together are diagnostic.
    func testDensePhaseDetectGridIsDiagnosedNotBlamedOnTheThreshold() throws {
        let width = 128, height = 128
        var plane = mosaic(width, height)
        for row in stride(from: 16, to: 112, by: 16) {
            for column in stride(from: 8, to: 120, by: 16) {
                plane[row * width + column] = 1.0
            }
        }
        XCTAssertThrowsError(try BadPixelCorrection.detectAndCorrect(
            cfa: plane, width: width, height: height)) { error in
            guard case BadPixelCorrection.CorrectionError.phaseDetectPatternDetected = error else {
                return XCTFail("expected a PDAF diagnosis, got \(error)")
            }
        }
    }

    /// Scattered defects must NOT trip the PDAF flag, or the signal is worthless.
    func testScatteredDefectsAreNotFlaggedAsPhaseDetect() throws {
        let width = 128, height = 128
        var plane = mosaic(width, height)
        var state: UInt32 = 0x1234
        for _ in 0..<20 {
            state = state &* 1_664_525 &+ 1_013_904_223
            let x = 4 + Int((state >> 8) % UInt32(width - 8))
            state = state &* 1_664_525 &+ 1_013_904_223
            let y = 4 + Int((state >> 8) % UInt32(height - 8))
            plane[y * width + x] = 1.0
        }
        let (_, report) = try BadPixelCorrection.detectAndCorrect(
            cfa: plane, width: width, height: height)
        XCTAssertFalse(report.phaseDetectPatternSuspected)
    }

    /// 🚨 **The safety valve.** At an implausible defect density this stops being repair and becomes a
    /// smoothing filter over every isolated detail in the frame — so it aborts instead.
    func testImplausibleDensityAbortsRatherThanSmoothing() throws {
        let width = 64, height = 64
        var state: UInt32 = 0x99
        let noise = (0..<(width * height)).map { _ -> Float in
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float((state >> 16) & 0xffff) / 65535
        }
        var reckless = BadPixelCorrection.Options.default
        reckless.threshold = 0.01                     // flag almost everything

        XCTAssertThrowsError(try BadPixelCorrection.detectAndCorrect(
            cfa: noise, width: width, height: height, options: reckless)) { error in
            guard case BadPixelCorrection.CorrectionError.implausibleDefectDensity = error else {
                return XCTFail("wrong error: \(error)")
            }
        }
    }

    func testCleanSensorIsUntouched() throws {
        let width = 64, height = 64
        let clean = mosaic(width, height)
        let (out, report) = try BadPixelCorrection.detectAndCorrect(
            cfa: clean, width: width, height: height)
        XCTAssertEqual(report.detected, 0)
        XCTAssertEqual(out, clean)
    }

    func testSizeGuards() {
        XCTAssertThrowsError(try BadPixelCorrection.detect(
            cfa: [Float](repeating: 0, count: 10), width: 5, height: 3)) {
            XCTAssertEqual($0 as? BadPixelCorrection.CorrectionError, .sizeMismatch)
        }
        XCTAssertThrowsError(try BadPixelCorrection.detect(
            cfa: [Float](repeating: 0, count: 9), width: 3, height: 3)) {
            XCTAssertEqual($0 as? BadPixelCorrection.CorrectionError, .tooSmall)
        }
    }
}
