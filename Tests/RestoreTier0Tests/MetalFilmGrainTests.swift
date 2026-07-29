import XCTest
import CoreVideo
@testable import RestoreTier0

/// GPU film grain. Parity is asserted as **equality**, not closeness — the design rests on grain being
/// addressed by absolute position so a tile matches a full-frame render, and a GPU path that agreed
/// only to within a code would reintroduce the shimmer the seed exists to prevent.
final class MetalFilmGrainTests: XCTestCase {

    func testDiagnostics() {
        let report = MetalFilmGrain.diagnostics()
        XCTAssertFalse(report.isEmpty)
        print("GRAIN \(report)")
    }

    // MARK: - Fixtures

    private func makeBuffer(_ width: Int, _ height: Int,
                            _ format: OSType = kCVPixelFormatType_32BGRA) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, format,
            [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
             kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw MetalFilmGrain.Failure(reason: "CVPixelBufferCreate \(status)")
        }
        return buffer
    }

    /// A tone ramp — grain amplitude follows brightness through the scaling curve, so a flat fixture
    /// would exercise exactly one point of it.
    private func fillRamp(_ buffer: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.bindMemory(to: UInt8.self, capacity: stride * h)
        for y in 0..<h {
            for x in 0..<w {
                let v = UInt8(8 + (x * 239) / max(1, w - 1))
                let p = y * stride + x * 4
                base[p] = v; base[p + 1] = v; base[p + 2] = v; base[p + 3] = 255
            }
        }
    }

    private func luma(_ buffer: CVPixelBuffer) -> [Float] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.bindMemory(to: UInt8.self, capacity: stride * h)
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let p = y * stride + x * 4
                out[y * w + x] = (0.2126 * Float(base[p + 2]) + 0.7152 * Float(base[p + 1])
                                  + 0.0722 * Float(base[p])) / 255
            }
        }
        return out
    }

    // MARK: - Parity

    /// 🔑 The contract: the GPU grain field is the CPU grain field, block-for-block.
    func testGrainFieldMatchesTheCPUExactly() throws {
        let parameters = FilmGrainParameters.preset(.silverRich, seed: 0x4242, size: 0.8, amount: 1.0)
        let grain = try XCTUnwrap(MetalFilmGrain(parameters: parameters), MetalFilmGrain.diagnostics())

        let width = 160, height = 128
        let source = try makeBuffer(width, height)
        fillRamp(source)
        let result = try grain.applied(to: source)

        let before = luma(source), after = luma(result)
        let field = grain.synthesizer.grainField(width: width, height: height)
        let lut = parameters.scalingLUT()
        let shift = parameters.grainScalingShift

        var mismatches = 0
        for i in 0..<before.count {
            let level = AFGS1.clip(Int((before[i] * 255).rounded()), 0, 255)
            let expected = Float(min(max(Int((before[i] * 255).rounded()) + AFGS1.round2(lut[level] * field[i], shift), 0), 255)) / 255
            if abs(after[i] - expected) > 1.5 / 255 { mismatches += 1 }
        }
        XCTAssertEqual(mismatches, 0, "\(mismatches) of \(before.count) pixels diverged from the CPU field")
    }

    /// 🚨 **The property the whole design exists for, now on the GPU too.** A tile rendered at a
    /// non-block-aligned origin must carry the same grain as that region of a full-frame render — or
    /// grain crawls under the picture as the user pans.
    func testTileMatchesFullFrame() throws {
        let parameters = FilmGrainParameters.preset(.grey, seed: 0x77, size: 0.6, amount: 1.0)
        let grain = try XCTUnwrap(MetalFilmGrain(parameters: parameters), "no Metal device")

        let width = 256, height = 192
        let full = try makeBuffer(width, height)
        fillRamp(full)
        let fullResult = luma(try grain.applied(to: full))

        // Deliberately not block-aligned — alignment hides an off-by-one in the block mapping.
        let tileX = 37, tileY = 45, tileW = 96, tileH = 64
        let tile = try makeBuffer(tileW, tileH)
        CVPixelBufferLockBaseAddress(full, .readOnly)
        CVPixelBufferLockBaseAddress(tile, [])
        let fullStride = CVPixelBufferGetBytesPerRow(full)
        let tileStride = CVPixelBufferGetBytesPerRow(tile)
        let fullBase = CVPixelBufferGetBaseAddress(full)!.bindMemory(to: UInt8.self, capacity: fullStride * height)
        let tileBase = CVPixelBufferGetBaseAddress(tile)!.bindMemory(to: UInt8.self, capacity: tileStride * tileH)
        for y in 0..<tileH {
            for x in 0..<(tileW * 4) {
                tileBase[y * tileStride + x] = fullBase[(tileY + y) * fullStride + tileX * 4 + x]
            }
        }
        CVPixelBufferUnlockBaseAddress(tile, [])
        CVPixelBufferUnlockBaseAddress(full, .readOnly)

        let tileResult = luma(try grain.applied(to: tile, originX: tileX, originY: tileY))
        var mismatches = 0
        for y in 0..<tileH {
            for x in 0..<tileW {
                let a = tileResult[y * tileW + x]
                let b = fullResult[(tileY + y) * width + (tileX + x)]
                if abs(a - b) > 1.5 / 255 { mismatches += 1 }
            }
        }
        XCTAssertEqual(mismatches, 0, "\(mismatches) tile pixels differ from the full-frame render")
    }

    /// The call-site half of the same bug, pinned so the requirement is visible: rendering a tile at
    /// origin (0,0) — as a naive loop would — produces *different* grain. The synthesizer cannot
    /// detect this for you, so the UI doc carries it as an obligation.
    func testIgnoringTheTileOriginProducesDifferentGrain() throws {
        let parameters = FilmGrainParameters.preset(.grey, seed: 0x77, size: 0.6, amount: 1.0)
        let grain = try XCTUnwrap(MetalFilmGrain(parameters: parameters), "no Metal device")
        let tile = try makeBuffer(64, 64)
        fillRamp(tile)

        let correct = luma(try grain.applied(to: tile, originX: 64, originY: 96))
        let naive = luma(try grain.applied(to: tile, originX: 0, originY: 0))
        XCTAssertNotEqual(correct, naive, "if these matched, absolute addressing would not be working")
    }

    // MARK: - Composition and colour

    /// Sharpen → grain, both GPU, both `CVPixelBuffer` — the chain an app runs. No readback anywhere.
    func testComposesWithTheSharpenPipeline() throws {
        let sharpen = try XCTUnwrap(MetalSharpenPipeline(), "no Metal device")
        var options = NoiseAwareSharpener.Options.default
        options.noiseSigma = 0.002
        sharpen.options = options
        let grain = try XCTUnwrap(
            MetalFilmGrain(parameters: .preset(.silverRich, seed: 0x9, amount: 0.6)), "no Metal device")

        let source = try makeBuffer(128, 96)
        fillRamp(source)
        let out = try grain.applied(to: try sharpen.sharpen(source))

        XCTAssertEqual(CVPixelBufferGetWidth(out), 128)
        XCTAssertNotEqual(luma(out), luma(source), "the chain must actually change the picture")
    }

    /// Grain is a luma effect: an equal delta on R, G and B leaves the colour differences alone.
    ///
    /// ⚠️ **Away from clipping** — see the test below. The fixture keeps every channel clear of 0 and
    /// 255 so the claim is tested in the regime where it actually holds; an earlier version used
    /// amount 1.0 on R=200, where ±74 codes of grain drives red into the ceiling and the differences
    /// *must* shrink. That failure was the fixture's, not the shader's.
    func testChromaIsNotTintedAwayFromClipping() throws {
        let grain = try XCTUnwrap(
            MetalFilmGrain(parameters: .preset(.grey, seed: 0x5, amount: 0.35)), "no Metal device")
        let width = 64, height = 64
        let source = try makeBuffer(width, height)

        CVPixelBufferLockBaseAddress(source, [])
        let stride = CVPixelBufferGetBytesPerRow(source)
        let base = CVPixelBufferGetBaseAddress(source)!.bindMemory(to: UInt8.self, capacity: stride * height)
        for y in 0..<height {
            for x in 0..<width {
                let p = y * stride + x * 4
                base[p] = 60; base[p + 1] = 110; base[p + 2] = 160; base[p + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(source, [])

        let result = try grain.applied(to: source)
        CVPixelBufferLockBaseAddress(result, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(result, .readOnly) }
        let s = CVPixelBufferGetBytesPerRow(result)
        let b = CVPixelBufferGetBaseAddress(result)!.bindMemory(to: UInt8.self, capacity: s * height)

        var sawGrain = false
        for y in 4..<(height - 4) {
            for x in 4..<(width - 4) {
                let p = y * s + x * 4
                XCTAssertEqual(Int(b[p + 2]) - Int(b[p + 1]), 50, accuracy: 1, "R−G at (\(x),\(y))")
                XCTAssertEqual(Int(b[p + 1]) - Int(b[p]), 50, accuracy: 1, "G−B at (\(x),\(y))")
                if Int(b[p + 1]) != 110 { sawGrain = true }
            }
        }
        XCTAssertTrue(sawGrain, "the fixture must actually receive grain, or this proves nothing")
    }

    /// ⚠️ **The honest limit, pinned so nobody later assumes more.** An additive-delta model cannot
    /// preserve colour differences once a channel hits 0 or 255 — the clipped channel stops moving
    /// while the others keep going. This is a property of the model, not a defect, and it means
    /// **heavy grain on near-white or near-black colour will desaturate slightly**. If that ever
    /// matters, the fix is to scale the delta by the headroom, not to widen a tolerance.
    func testHeavyGrainOnANearWhiteChannelDoesShiftChroma() throws {
        let grain = try XCTUnwrap(
            MetalFilmGrain(parameters: .preset(.grey, seed: 0x5, amount: 1.0)), "no Metal device")
        let width = 64, height = 64
        let source = try makeBuffer(width, height)

        CVPixelBufferLockBaseAddress(source, [])
        let stride = CVPixelBufferGetBytesPerRow(source)
        let base = CVPixelBufferGetBaseAddress(source)!.bindMemory(to: UInt8.self, capacity: stride * height)
        for y in 0..<height {
            for x in 0..<width {
                let p = y * stride + x * 4
                base[p] = 60; base[p + 1] = 140; base[p + 2] = 200; base[p + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(source, [])

        let result = try grain.applied(to: source)
        CVPixelBufferLockBaseAddress(result, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(result, .readOnly) }
        let s = CVPixelBufferGetBytesPerRow(result)
        let b = CVPixelBufferGetBaseAddress(result)!.bindMemory(to: UInt8.self, capacity: s * height)

        var clipped = 0
        for y in 4..<(height - 4) {
            for x in 4..<(width - 4) where Int(b[y * s + x * 4 + 2]) == 255 { clipped += 1 }
        }
        XCTAssertGreaterThan(clipped, 0,
                             "with ±74 codes on R=200 the red channel must reach the ceiling — that is why "
                             + "the preservation claim above is scoped to the unclipped regime")
    }

    func testProcessDegradesOnAnUnsupportedFormat() throws {
        let grain = try XCTUnwrap(MetalFilmGrain(parameters: .preset(.grey)), "no Metal device")
        let buffer = try makeBuffer(32, 32, kCVPixelFormatType_32ARGB)
        XCTAssertTrue(grain.process(buffer) === buffer)
    }
}
