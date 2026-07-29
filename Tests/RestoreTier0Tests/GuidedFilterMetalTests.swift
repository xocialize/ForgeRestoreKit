import XCTest
import Metal
@testable import RestoreTier0

/// The GPU backend must be a *drop-in* — same answers, faster. Parity is the whole contract, so it is
/// tested rather than assumed, and the CPU implementation is the oracle.
final class GuidedFilterMetalTests: XCTestCase {

    /// If this fails, the message says whether it is a missing device, a runtime shader-compile
    /// failure, or a missing kernel — three problems with three different fixes.
    func testDiagnosticsReportsAvailability() {
        let report = GuidedFilterMetal.diagnostics()
        XCTAssertFalse(report.isEmpty)
        print("METAL \(report)")
    }

    /// 🔑 **Parity, on content with real structure at several scales.** A flat or trivial fixture would
    /// pass with a broken `a`/`b` stage, since `q` collapses to the mean either way.
    func testMatchesTheCPUFilter() throws {
        let metal = try XCTUnwrap(GuidedFilterMetal.shared, "no Metal device: \(GuidedFilterMetal.diagnostics())")
        let width = 129, height = 97          // deliberately not multiples of the threadgroup size
        var state: UInt32 = 0xA11CE
        let plane = (0..<(width * height)).map { i -> Float in
            state = state &* 1_664_525 &+ 1_013_904_223
            let x = Float(i % width), y = Float(i / width)
            let structure = 0.5 + 0.25 * Foundation.sin(x / 9) * Foundation.cos(y / 13)
            return structure + (Float((state >> 16) & 0xffff) / 65535 - 0.5) * 0.05
        }

        let cpu = GuidedFilter()
        for (radius, epsilon) in [(2, Float(0.0025)), (8, 0.0025), (32, 0.01)] {
            let expected = cpu.base(of: plane, width: width, height: height,
                                    radius: radius, epsilon: epsilon)
            let got = metal.base(of: plane, width: width, height: height,
                                 radius: radius, epsilon: epsilon)
            XCTAssertEqual(got.count, expected.count)

            var maxError: Float = 0
            for i in 0..<expected.count { maxError = max(maxError, abs(got[i] - expected[i])) }
            // Tolerance covers the rolling-vs-fresh summation difference, NOT a shortcut: at 8-bit a
            // code is 1/255 ≈ 3.9e-3, so 1e-4 is ~40× inside the smallest visible step.
            XCTAssertLessThan(maxError, 1e-4,
                              "radius \(radius) ε \(epsilon): maxError \(maxError)")
        }
    }

    /// The degenerate case both implementations must agree on, and the one that used to produce NaN.
    func testFlatNoiselessRegionAgreesAndIsNotNaN() throws {
        let metal = try XCTUnwrap(GuidedFilterMetal.shared, "no Metal device")
        let plane = [Float](repeating: 0.5, count: 64 * 64)
        let got = metal.base(of: plane, width: 64, height: 64, radius: 4, epsilon: 0)
        for v in got {
            XCTAssertFalse(v.isNaN, "0/0 must resolve to a = 0, not NaN")
            XCTAssertEqual(v, 0.5, accuracy: 1e-5)
        }
    }

    /// The integration that matters: swapping the backend must not change what the sharpener does.
    func testSharpenerAgreesAcrossBackends() throws {
        let metal = try XCTUnwrap(GuidedFilterMetal.shared, "no Metal device")
        let width = 96, height = 96
        let plane = (0..<(width * height)).map { i -> Float in
            0.5 + 0.03 * Foundation.sin(Float(i % width) * 2 * .pi / 6)
        }

        var options = NoiseAwareSharpener.Options.default
        options.noiseSigma = 0.0005
        let onCPU = NoiseAwareSharpener(options: options, filter: GuidedFilter())
            .sharpen(plane, width: width, height: height)
        let onGPU = NoiseAwareSharpener(options: options, filter: metal)
            .sharpen(plane, width: width, height: height)

        var maxError: Float = 0
        for i in 0..<plane.count { maxError = max(maxError, abs(onCPU.plane[i] - onGPU.plane[i])) }
        XCTAssertLessThan(maxError, 1e-3, "backend swap changed the result by \(maxError)")
        XCTAssertEqual(onCPU.report.appliedBandDiameters, onGPU.report.appliedBandDiameters)
    }
}
