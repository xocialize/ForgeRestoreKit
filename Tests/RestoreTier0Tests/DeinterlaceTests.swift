import XCTest
@testable import RestoreTier0

/// N9 — deinterlacing. The claim worth proving is the structural one: where source field lines are
/// kept they are **provably the original samples**, not a reconstruction that scores well. That is what
/// no learned method can offer, and it is asserted bit-exactly.
final class DeinterlaceTests: XCTestCase {

    // MARK: - Fixtures

    /// Weave two moments together: even rows from `a`, odd rows from `b` — an interlaced frame.
    private func weave(_ a: [Float], _ b: [Float], width: Int, height: Int) -> [Float] {
        var out = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let source = (y % 2 == 0) ? a : b
            for x in 0..<width { out[y * width + x] = source[y * width + x] }
        }
        return out
    }

    /// A diagonal edge — the case edge-directed averaging exists for.
    private func diagonal(_ width: Int, _ height: Int, shift: Float = 0) -> [Float] {
        (0..<(width * height)).map { i in
            let x = Float(i % width), y = Float(i / width)
            return (x + shift) > y * 3 ? 0.85 : 0.15
        }
    }

    private func flat(_ width: Int, _ height: Int, _ level: Float) -> [Float] {
        [Float](repeating: level, count: width * height)
    }

    // MARK: - The structural claim

    /// 🔑 **`lossless` means bit-exact, not "close".** Every retained field line must survive
    /// unmodified — this is the property that makes classical structurally unbeatable for archival
    /// work, and the one a learned method cannot promise.
    func testLosslessKeepsSourceFieldLinesBitExact() throws {
        let width = 64, height = 32
        let frame = weave(diagonal(width, height), diagonal(width, height, shift: 9),
                          width: width, height: height)
        let out = try Deinterlace.frame(current: frame, width: width, height: height, parity: .top)

        for y in stride(from: 0, to: height, by: 2) {
            for x in 0..<width {
                XCTAssertEqual(out[y * width + x], frame[y * width + x],
                               "top-field row \(y) col \(x) was modified")
            }
        }
    }

    /// …and the missing lines must actually be replaced, or "lossless" would just mean "did nothing".
    func testMissingLinesAreReconstructed() throws {
        let width = 64, height = 32
        let frame = weave(diagonal(width, height), diagonal(width, height, shift: 9),
                          width: width, height: height)
        let out = try Deinterlace.frame(current: frame, width: width, height: height, parity: .top)

        var changed = 0
        for y in stride(from: 1, to: height, by: 2) {
            for x in 0..<width where out[y * width + x] != frame[y * width + x] { changed += 1 }
        }
        XCTAssertGreaterThan(changed, width, "the opposite-parity lines must be rebuilt")
    }

    func testBottomFieldParityKeepsTheOtherLines() throws {
        let width = 48, height = 24
        let frame = weave(flat(width, height, 0.2), flat(width, height, 0.8),
                          width: width, height: height)
        let out = try Deinterlace.frame(current: frame, width: width, height: height, parity: .bottom)
        for y in stride(from: 1, to: height, by: 2) {
            for x in 0..<width { XCTAssertEqual(out[y * width + x], 0.8) }
        }
    }

    /// Turning `lossless` off must genuinely let the source lines move — otherwise the flag is a lie.
    func testLosslessOffAllowsSourceLinesToChange() throws {
        let width = 48, height = 24
        let frame = weave(diagonal(width, height), diagonal(width, height, shift: 7),
                          width: width, height: height)
        var options = Deinterlace.Options.default
        options.lossless = false
        let out = try Deinterlace.frame(current: frame, width: width, height: height,
                                        parity: .top, options: options)
        var changed = 0
        for y in stride(from: 0, to: height, by: 2) {
            for x in 0..<width where out[y * width + x] != frame[y * width + x] { changed += 1 }
        }
        XCTAssertGreaterThan(changed, 0)
    }

    // MARK: - Edge-directed interpolation

    /// 🔑 **The reason not to average vertically.** On a near-horizontal edge a vertical average blends
    /// both sides and staircases; following the edge keeps it intact. Measured as how many
    /// reconstructed samples land in the ambiguous middle instead of committing to a side.
    func testEdgeDirectedBeatsVerticalAveragingOnADiagonal() throws {
        let width = 96, height = 48
        let frame = diagonal(width, height)

        func middling(_ search: Int) throws -> Int {
            var options = Deinterlace.Options.default
            options.mode = .bob
            options.edgeSearch = search
            let out = try Deinterlace.frame(current: frame, width: width, height: height,
                                            parity: .top, options: options)
            var count = 0
            for y in stride(from: 1, to: height, by: 2) {
                for x in 0..<width {
                    let v = out[y * width + x]
                    if v > 0.3 && v < 0.7 { count += 1 }     // neither side of the edge
                }
            }
            return count
        }
        let vertical = try middling(0)      // search 0 == plain vertical average
        let directed = try middling(3)
        XCTAssertGreaterThan(vertical, 0, "the fixture must actually have a slanted edge")
        XCTAssertLessThan(directed, vertical,
                          "edge-directed left \(directed) ambiguous samples vs vertical's \(vertical)")
    }

    /// A flat region must fall back to the vertical average rather than picking a slanted direction —
    /// that bias is what stops ELA streaking on noise.
    func testFlatRegionsUseTheVerticalAverage() throws {
        let width = 48, height = 24
        let frame = weave(flat(width, height, 0.4), flat(width, height, 0.6),
                          width: width, height: height)
        var options = Deinterlace.Options.default
        options.mode = .bob
        let out = try Deinterlace.frame(current: frame, width: width, height: height,
                                        parity: .top, options: options)
        for y in stride(from: 3, to: height - 2, by: 2) {
            for x in 2..<(width - 2) {
                XCTAssertEqual(out[y * width + x], 0.4, accuracy: 1e-5)
            }
        }
    }

    // MARK: - Motion adaptation

    /// 🔑 **Weave is EXACT where nothing moved** — it is not an approximation there, it is the original
    /// samples of the other field. So a static scene must reconstruct essentially perfectly, and far
    /// better than the spatial path alone could.
    ///
    /// ⚠️ The fixture must be **vertical detail at Nyquist** — alternating lines. A diagonal edge is a
    /// bad test here and the first version used one: ELA reconstructs a smooth slant *exactly*, so both
    /// paths scored 0 and the comparison was vacuous. Line-alternating content is the case no spatial
    /// interpolator can recover (the two neighbours are identical, so any interpolation returns the
    /// wrong value) and that weave recovers perfectly when nothing moved.
    func testStaticSceneReconstructsFromTheTemporalPath() throws {
        let width = 64, height = 32
        // Alternating lines: spatially unrecoverable, temporally exact.
        let truth = (0..<(width * height)).map { i -> Float in ((i / width) % 2 == 0) ? 0.2 : 0.8 }
        // Nothing moves: all three frames are the same progressive picture.
        let interlaced = truth

        let motionAdaptive = try Deinterlace.frame(current: interlaced, previous: truth, next: truth,
                                                   width: width, height: height, parity: .top)
        var spatialOnly = Deinterlace.Options.default
        spatialOnly.mode = .bob
        let bobbed = try Deinterlace.frame(current: interlaced, width: width, height: height,
                                           parity: .top, options: spatialOnly)

        func error(_ plane: [Float]) -> Double {
            var total = 0.0
            for y in stride(from: 1, to: height, by: 2) {
                for x in 0..<width { total += abs(Double(plane[y * width + x] - truth[y * width + x])) }
            }
            return total
        }
        XCTAssertLessThan(error(motionAdaptive), error(bobbed) * 0.5,
                          "static: temporal \(error(motionAdaptive)) vs spatial \(error(bobbed))")
    }

    /// …and where there IS motion the temporal estimate must not be trusted, or it ghosts.
    func testMotionSuppressesTheTemporalPath() throws {
        let width = 64, height = 32
        let current = diagonal(width, height)
        let moved = diagonal(width, height, shift: 24)      // a large displacement

        let out = try Deinterlace.frame(current: current, previous: moved, next: moved,
                                        width: width, height: height, parity: .top)
        // The reconstruction must stay inside its vertical neighbourhood rather than being dragged
        // toward the displaced frame — that drag is exactly what a ghost is.
        for y in stride(from: 3, to: height - 2, by: 2) {
            for x in 1..<(width - 1) {
                let low = min(current[(y - 1) * width + x], current[(y + 1) * width + x])
                let high = max(current[(y - 1) * width + x], current[(y + 1) * width + x])
                let slack = (high - low) * 0.5 + Deinterlace.Options.default.motionThreshold + 1e-4
                XCTAssertGreaterThanOrEqual(out[y * width + x], low - slack, "ghost low at (\(x),\(y))")
                XCTAssertLessThanOrEqual(out[y * width + x], high + slack, "ghost high at (\(x),\(y))")
            }
        }
    }

    /// No temporal input must degrade to bob *and say so by behaving identically* — not silently
    /// produce something worse under the motion-adaptive name.
    func testNoTemporalInputMatchesBob() throws {
        let width = 48, height = 24
        let frame = weave(diagonal(width, height), diagonal(width, height, shift: 5),
                          width: width, height: height)
        let adaptive = try Deinterlace.frame(current: frame, width: width, height: height, parity: .top)
        var options = Deinterlace.Options.default
        options.mode = .bob
        let bobbed = try Deinterlace.frame(current: frame, width: width, height: height,
                                           parity: .top, options: options)
        XCTAssertEqual(adaptive, bobbed)
    }

    // MARK: - Bob

    func testBobDoublesHeightAndPlacesTheFieldCorrectly() throws {
        let width = 32, fieldHeight = 12
        let field = (0..<(width * fieldHeight)).map { Float($0 % width) / Float(width) }
        let out = try Deinterlace.bob(field: field, fieldWidth: width, fieldHeight: fieldHeight,
                                      parity: .top)
        XCTAssertEqual(out.count, width * fieldHeight * 2)
        for row in 0..<fieldHeight {
            for x in 0..<width {
                XCTAssertEqual(out[(row * 2) * width + x], field[row * width + x],
                               "the field's own samples must survive at row \(row * 2)")
            }
        }
    }

    // MARK: - Comb detection

    /// 🚨 Field-order errors and progressive-in-an-interlaced-container are both silent in a pipeline.
    /// A woven frame of two different moments must register; a progressive frame must not.
    func testCombDetectorSeparatesWovenFromProgressive() {
        let width = 64, height = 48
        let woven = weave(diagonal(width, height), diagonal(width, height, shift: 12),
                          width: width, height: height)
        let progressive = diagonal(width, height)

        let combed = CombDetector.detect(woven, width: width, height: height)
        let clean = CombDetector.detect(progressive, width: width, height: height)

        XCTAssertTrue(combed.isInterlaced, "combing energy \(combed.combEnergy)")
        XCTAssertFalse(clean.isInterlaced, "progressive energy \(clean.combEnergy)")
        XCTAssertGreaterThan(combed.combEnergy, clean.combEnergy * 4)
    }

    /// A smooth gradient must not read as combing — the `(a−b)(c−b)` product is *negative* there, which
    /// is precisely why that measure is used rather than a plain row difference.
    func testSmoothGradientIsNotMistakenForCombing() {
        let width = 64, height = 48
        let ramp = (0..<(width * height)).map { Float($0 / width) / Float(height) }
        XCTAssertFalse(CombDetector.detect(ramp, width: width, height: height).isInterlaced)
    }

    /// Deinterlacing must reduce comb energy — the end-to-end claim.
    func testDeinterlacingReducesCombEnergy() throws {
        let width = 64, height = 48
        let woven = weave(diagonal(width, height), diagonal(width, height, shift: 12),
                          width: width, height: height)
        let out = try Deinterlace.frame(current: woven, width: width, height: height, parity: .top)
        let before = CombDetector.detect(woven, width: width, height: height).combEnergy
        let after = CombDetector.detect(out, width: width, height: height).combEnergy
        XCTAssertLessThan(after, before * 0.5, "comb energy \(before) → \(after)")
    }

    // MARK: - Guards

    func testSizeGuards() {
        XCTAssertThrowsError(try Deinterlace.frame(current: [Float](repeating: 0, count: 10),
                                                   width: 5, height: 4, parity: .top)) {
            XCTAssertEqual($0 as? Deinterlace.DeinterlaceError, .sizeMismatch)
        }
        XCTAssertThrowsError(try Deinterlace.frame(current: [Float](repeating: 0, count: 4),
                                                   width: 2, height: 2, parity: .top)) {
            XCTAssertEqual($0 as? Deinterlace.DeinterlaceError, .tooSmall)
        }
    }
}
