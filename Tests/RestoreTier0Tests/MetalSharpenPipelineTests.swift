import XCTest
import CoreVideo
import Metal
@testable import RestoreTier0

/// The GPU-resident chain. Parity against the CPU oracle is the contract; everything else here guards
/// the two things that make a pixel-buffer path go quietly wrong — colour handling and layout.
final class MetalSharpenPipelineTests: XCTestCase {

    func testDiagnostics() {
        let report = MetalSharpenPipeline.diagnostics()
        XCTAssertFalse(report.isEmpty)
        print("PIPELINE \(report)")
    }

    // MARK: - Fixtures

    private func makeBuffer(width: Int, height: Int, format: OSType) throws -> CVPixelBuffer {
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            kCVPixelBufferMetalCompatibilityKey: true,
        ]
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format,
                                         attributes as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw MetalSharpenPipeline.Failure(reason: "CVPixelBufferCreate \(status)")
        }
        return buffer
    }

    /// Texture: fine detail plus a hard edge, so both the boosted case and the protected case appear.
    private func lumaValue(_ x: Int, _ y: Int, _ width: Int) -> Float {
        let base: Float = x < width / 2 ? 0.35 : 0.62
        return base + 0.03 * Foundation.sin(Float(x) * 2 * .pi / 6) * Foundation.cos(Float(y) / 11)
    }

    private func fillBGRA(_ buffer: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.bindMemory(to: UInt8.self, capacity: stride * h)
        for y in 0..<h {
            for x in 0..<w {
                let v = UInt8(max(0, min(255, (lumaValue(x, y, w) * 255).rounded())))
                let p = y * stride + x * 4
                base[p] = v; base[p + 1] = v; base[p + 2] = v; base[p + 3] = 255   // BGRA, grey
            }
        }
    }

    private func plane(from buffer: CVPixelBuffer) -> [Float] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.bindMemory(to: UInt8.self, capacity: stride * h)
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let p = y * stride + x * 4
                let b = Float(base[p]), g = Float(base[p + 1]), r = Float(base[p + 2])
                out[y * w + x] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
            }
        }
        return out
    }

    // MARK: - Parity

    /// 🔑 **The contract.** The GPU chain must agree with the CPU reference. σ is supplied so the two
    /// share an identical noise floor — the estimator itself is covered separately.
    func testGPUChainMatchesTheCPUReference() throws {
        let pipeline = try XCTUnwrap(MetalSharpenPipeline(), MetalSharpenPipeline.diagnostics())
        let width = 128, height = 96
        let source = try makeBuffer(width: width, height: height, format: kCVPixelFormatType_32BGRA)
        fillBGRA(source)

        var options = NoiseAwareSharpener.Options.default
        options.noiseSigma = 0.002
        pipeline.options = options

        let result = try pipeline.sharpen(source)
        let got = plane(from: result)

        let reference = NoiseAwareSharpener(options: options)
            .sharpen(plane(from: source), width: width, height: height).plane

        var maxError: Float = 0
        for i in 0..<got.count { maxError = max(maxError, abs(got[i] - reference[i])) }
        // The buffer round-trips through 8-bit, so one code (1/255 ≈ 3.9e-3) is the floor on agreement.
        XCTAssertLessThan(maxError, 2.5 / 255,
                          "GPU vs CPU maxError \(maxError) (\(maxError * 255) codes)")
    }

    /// The estimator must land in the same place on both paths, or every σ-derived constant diverges.
    func testMeasuredSigmaMatchesTheCPUEstimator() throws {
        let pipeline = try XCTUnwrap(MetalSharpenPipeline(), "no Metal device")
        let width = 96, height = 96
        let source = try makeBuffer(width: width, height: height, format: kCVPixelFormatType_32BGRA)
        fillBGRA(source)

        let destination = try XCTUnwrap(PixelBufferBridge.makeCompatibleBuffer(like: source))
        let report = try pipeline.run(source: source, destination: destination)
        let expected = NoiseEstimate.immerkaer(plane(from: source), width: width, height: height)

        XCTAssertFalse(report.sigmaWasSupplied)
        XCTAssertEqual(report.noiseSigma, expected, accuracy: 1e-4,
                       "GPU σ \(report.noiseSigma) vs CPU \(expected)")
    }

    // MARK: - Colour and layout

    /// 🚨 **Chroma must survive.** The luma delta is applied equally to R, G and B, so the colour
    /// differences (R−Y, B−Y) are unchanged. A saturated input that comes back desaturated — or
    /// hue-shifted — means the write-back is wrong, and it is the kind of thing that looks like
    /// "the sharpener is too strong" rather than a bug.
    func testChromaIsPreservedOnColourContent() throws {
        let pipeline = try XCTUnwrap(MetalSharpenPipeline(), "no Metal device")
        let width = 64, height = 64
        let source = try makeBuffer(width: width, height: height, format: kCVPixelFormatType_32BGRA)

        CVPixelBufferLockBaseAddress(source, [])
        let stride = CVPixelBufferGetBytesPerRow(source)
        let base = CVPixelBufferGetBaseAddress(source)!.bindMemory(to: UInt8.self, capacity: stride * height)
        for y in 0..<height {
            for x in 0..<width {
                let p = y * stride + x * 4
                let wobble = Int(6 * Foundation.sin(Float(x) * 2 * .pi / 5))
                base[p] = UInt8(max(0, min(255, 40 + wobble)))        // B
                base[p + 1] = UInt8(max(0, min(255, 150 + wobble)))   // G
                base[p + 2] = UInt8(max(0, min(255, 210 + wobble)))   // R
                base[p + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(source, [])

        var options = NoiseAwareSharpener.Options.default
        options.noiseSigma = 0.002
        pipeline.options = options
        let result = try pipeline.sharpen(source)

        func differences(_ buffer: CVPixelBuffer) -> (rMinusG: Float, bMinusG: Float) {
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            let s = CVPixelBufferGetBytesPerRow(buffer)
            let b = CVPixelBufferGetBaseAddress(buffer)!.bindMemory(to: UInt8.self, capacity: s * height)
            var rg = 0.0, bg = 0.0, n = 0.0
            for y in 8..<(height - 8) {
                for x in 8..<(width - 8) {
                    let p = y * s + x * 4
                    rg += Double(Int(b[p + 2]) - Int(b[p + 1]))
                    bg += Double(Int(b[p]) - Int(b[p + 1]))
                    n += 1
                }
            }
            return (Float(rg / n), Float(bg / n))
        }

        let before = differences(source), after = differences(result)
        XCTAssertEqual(after.rMinusG, before.rMinusG, accuracy: 1.5, "R−G drifted")
        XCTAssertEqual(after.bMinusG, before.bMinusG, accuracy: 1.5, "B−G drifted")
    }

    /// Biplanar is the easy case for the same reason it is the honest one: plane 1 is never bound, so
    /// chroma cannot change. This asserts it byte-for-byte.
    func testBiplanarChromaPlaneIsUntouched() throws {
        let pipeline = try XCTUnwrap(MetalSharpenPipeline(), "no Metal device")
        let width = 64, height = 64
        let source = try makeBuffer(width: width, height: height,
                                    format: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)

        CVPixelBufferLockBaseAddress(source, [])
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(source, 0)
        let y0 = CVPixelBufferGetBaseAddressOfPlane(source, 0)!.bindMemory(to: UInt8.self, capacity: yStride * height)
        for y in 0..<height {
            for x in 0..<width {
                y0[y * yStride + x] = UInt8(max(0, min(255, (lumaValue(x, y, width) * 255).rounded())))
            }
        }
        let cStride = CVPixelBufferGetBytesPerRowOfPlane(source, 1)
        let cHeight = CVPixelBufferGetHeightOfPlane(source, 1)
        let c0 = CVPixelBufferGetBaseAddressOfPlane(source, 1)!.bindMemory(to: UInt8.self, capacity: cStride * cHeight)
        for i in 0..<(cStride * cHeight) { c0[i] = UInt8(90 + (i % 40)) }
        var chromaBefore = [UInt8](repeating: 0, count: cStride * cHeight)
        for i in 0..<chromaBefore.count { chromaBefore[i] = c0[i] }
        CVPixelBufferUnlockBaseAddress(source, [])

        var options = NoiseAwareSharpener.Options.default
        options.noiseSigma = 0.002
        pipeline.options = options
        let result = try pipeline.sharpen(source)
        XCTAssertEqual(try PixelBufferBridge.Layout.detect(result), .biplanarLuma)

        CVPixelBufferLockBaseAddress(result, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(result, .readOnly) }
        let outStride = CVPixelBufferGetBytesPerRowOfPlane(result, 1)
        let out = CVPixelBufferGetBaseAddressOfPlane(result, 1)!.bindMemory(to: UInt8.self, capacity: outStride * cHeight)
        for row in 0..<cHeight {
            for x in 0..<min(cStride, outStride) {
                XCTAssertEqual(out[row * outStride + x], chromaBefore[row * cStride + x],
                               "chroma changed at (\(x), \(row))")
            }
        }
    }

    func testUnsupportedFormatIsReportedNotGuessed() throws {
        let buffer = try makeBuffer(width: 32, height: 32, format: kCVPixelFormatType_32ARGB)
        XCTAssertThrowsError(try PixelBufferBridge.Layout.detect(buffer)) {
            guard case PixelBufferBridge.BridgeError.unsupportedPixelFormat = $0 else {
                return XCTFail("wrong error: \($0)")
            }
        }
    }

    /// `process` is the FrameProcessor shape: it must degrade, never trap.
    func testProcessReturnsTheInputOnAnUnsupportedFormat() throws {
        let pipeline = try XCTUnwrap(MetalSharpenPipeline(), "no Metal device")
        let buffer = try makeBuffer(width: 32, height: 32, format: kCVPixelFormatType_32ARGB)
        XCTAssertTrue(pipeline.process(buffer) === buffer)
    }
}
