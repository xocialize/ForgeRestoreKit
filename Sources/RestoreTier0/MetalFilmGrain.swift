//
// MetalFilmGrain.swift — RestoreTier0
//
// N2 on the GPU, entered and left as a `CVPixelBuffer` — so grain composes with
// `MetalSharpenPipeline` without ever leaving the GPU. Both take and return IOSurface-backed buffers,
// which the texture cache maps zero-copy, so chaining them costs an intermediate buffer and **no
// readback**. (The pessimization to avoid was always the `[Float]`-plane API, not `CVPixelBuffer`.)
//
// 🔑 **The split of labour is the plan's own, and it is what makes this cheap.** The AR filter that
// builds the 82×73 template is ~144k multiply-adds *once per image* — tens of microseconds in scalar
// Swift, and libplacebo, dav1d and FFmpeg all generate on the CPU for the same reason. The LFSR walk
// that assigns each block its offsets is **inherently sequential** (raster order, per-row reseed), so
// it stays on the CPU too. What ships to the GPU is the result: a small template, a 256-entry curve,
// and one `(offsetX, offsetY)` pair per block. The per-pixel stage is then a pure lookup —
// **no atomics, no barriers**, exactly as §N2 predicted.
//
// 🚨 **Parity here is bit-exact, not approximate, and that is deliberate.** The whole point of the
// design is that grain is addressed by *absolute* position so a viewport tile matches a full-frame
// render — a GPU path that agreed only to within a code would reintroduce the shimmer on pan that the
// seed exists to prevent. So the shader does the same **integer** arithmetic as the CPU (`Round2` as
// an arithmetic shift, the same clips), and the test asserts equality, not closeness.
//

import CoreVideo
import ForgePixelBridge
import Foundation
import Metal

public final class MetalFilmGrain: @unchecked Sendable {

    public struct Failure: Error, Equatable {
        public let reason: String
    }

    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let cache: CVMetalTextureCache
    private let applyPipeline: MTLComputePipelineState

    public let synthesizer: FilmGrainSynthesizer
    private let templateBuffer: MTLBuffer
    private let scalingBuffer: MTLBuffer

    public init?(parameters: FilmGrainParameters) {
        guard let synthesizer = try? FilmGrainSynthesizer(parameters: parameters),
              let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = try? device.makeLibrary(source: Self.kernelSource, options: nil),
              let function = library.makeFunction(name: "fg_apply"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let cache = try? PixelBufferBridge.makeTextureCache(device: device)
        else { return nil }

        // Template and curve are per-parameter-set, not per-frame — built once, uploaded once.
        let template = synthesizer.template.samples.map { Int32($0) }
        let curve = parameters.scalingLUT().map { Int32($0) }
        guard let templateBuffer = device.makeBuffer(
                bytes: template, length: template.count * MemoryLayout<Int32>.stride,
                options: .storageModeShared),
              let scalingBuffer = device.makeBuffer(
                bytes: curve, length: curve.count * MemoryLayout<Int32>.stride,
                options: .storageModeShared)
        else { return nil }

        self.device = device
        self.queue = queue
        self.cache = cache
        self.applyPipeline = pipeline
        self.synthesizer = synthesizer
        self.templateBuffer = templateBuffer
        self.scalingBuffer = scalingBuffer
    }

    public static func diagnostics(parameters: FilmGrainParameters = .preset(.grey)) -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "FAIL: no Metal device" }
        do {
            let library = try device.makeLibrary(source: kernelSource, options: nil)
            guard library.makeFunction(name: "fg_apply") != nil else {
                return "FAIL: \(device.name): missing kernel fg_apply"
            }
            return "OK \(device.name): film-grain kernel available"
        } catch {
            return "FAIL: \(device.name): makeLibrary(source:) — \(error.localizedDescription)"
        }
    }

    // MARK: - Entry points

    /// Matches `ImageBridge.FrameProcessor`. Degrades to the input on failure rather than trapping.
    public func process(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer {
        (try? applied(to: pixelBuffer)) ?? pixelBuffer
    }

    public func applied(to pixelBuffer: CVPixelBuffer,
                        originX: Int = 0, originY: Int = 0) throws -> CVPixelBuffer {
        guard let destination = PixelBufferBridge.makeCompatibleBuffer(like: pixelBuffer) else {
            throw Failure(reason: "could not allocate a destination buffer")
        }
        try run(source: pixelBuffer, destination: destination, originX: originX, originY: originY)
        return destination
    }

    /// Apply grain from `source` into `destination`.
    ///
    /// 🔑 **`originX`/`originY` are where this extent sits in the FULL picture.** A tiled renderer must
    /// pass the tile's true origin; passing `(0, 0)` for every tile is the call-site half of the
    /// shimmer-on-pan bug, and the synthesizer cannot detect it for you.
    public func run(source: CVPixelBuffer, destination: CVPixelBuffer,
                    originX: Int = 0, originY: Int = 0) throws {
        let layout = try PixelBufferBridge.Layout.detect(source)
        guard try PixelBufferBridge.Layout.detect(destination) == layout else {
            throw Failure(reason: "source and destination layouts differ")
        }
        let format = PixelBufferBridge.pixelFormat(for: layout, pixelBuffer: source)

        let (sourceWrapper, sourceTexture) = try PixelBufferBridge.texture(
            from: source, plane: 0, pixelFormat: format, cache: cache)
        let (destinationWrapper, destinationTexture) = try PixelBufferBridge.texture(
            from: destination, plane: 0, pixelFormat: format, cache: cache)
        defer { withExtendedLifetime((sourceWrapper, destinationWrapper)) {} }

        let width = sourceTexture.width, height = sourceTexture.height
        guard width > 0, height > 0 else { throw Failure(reason: "empty picture") }

        // The LFSR walk — sequential by construction, so it belongs here and not in a kernel.
        let size = AFGS1.blockSize
        let firstBlockRow = originY / size
        let lastBlockRow = (originY + height - 1) / size
        let lastBlockColumn = (originX + width - 1) / size
        let columns = lastBlockColumn + 1
        let neighbourRow = max(0, firstBlockRow - 1)

        var offsets: [SIMD2<Int32>] = []
        offsets.reserveCapacity((lastBlockRow - neighbourRow + 1) * columns)
        for row in neighbourRow...lastBlockRow {
            for offset in synthesizer.blockOffsets(blockRow: row, throughColumn: lastBlockColumn) {
                offsets.append(SIMD2<Int32>(Int32(offset.x), Int32(offset.y)))
            }
        }
        guard let offsetBuffer = device.makeBuffer(
                bytes: offsets, length: offsets.count * MemoryLayout<SIMD2<Int32>>.stride,
                options: .storageModeShared),
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder()
        else { throw Failure(reason: "could not encode the grain pass") }

        var config = Config(
            width: UInt32(width), height: UInt32(height),
            originX: Int32(originX), originY: Int32(originY),
            columns: Int32(columns), firstOffsetRow: Int32(neighbourRow),
            scalingShift: Int32(synthesizer.parameters.grainScalingShift),
            overlap: synthesizer.parameters.overlap ? 1 : 0,
            isLuma: layout == .biplanarLuma ? 1 : 0)

        encoder.setComputePipelineState(applyPipeline)
        encoder.setTexture(sourceTexture, index: 0)
        encoder.setTexture(destinationTexture, index: 1)
        encoder.setBuffer(templateBuffer, offset: 0, index: 0)
        encoder.setBuffer(scalingBuffer, offset: 0, index: 1)
        encoder.setBuffer(offsetBuffer, offset: 0, index: 2)
        encoder.setBytes(&config, length: MemoryLayout<Config>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw Failure(reason: "GPU: \(error.localizedDescription)") }
    }

    private struct Config {
        var width: UInt32
        var height: UInt32
        var originX: Int32
        var originY: Int32
        var columns: Int32
        var firstOffsetRow: Int32
        var scalingShift: Int32
        var overlap: Int32
        var isLuma: Int32
    }

    // MARK: - Kernel

    static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    constant float3 kLumaWeights = float3(0.2126f, 0.7152f, 0.0722f);
    constant int kTemplateWidth = 82;
    constant int kBlockSize = 32;
    constant int kSampleOrigin = 9;
    constant int kOverlapShift = 5;
    constant int kGrainMin = -128;
    constant int kGrainMax = 127;
    // Luma overlap weights, exactly 27/17 then 17/27.
    constant int2 kOverlapWeights[2] = { int2(27, 17), int2(17, 27) };

    struct Config {
        uint width;
        uint height;
        int originX;
        int originY;
        int columns;
        int firstOffsetRow;
        int scalingShift;
        int overlap;
        int isLuma;
    };

    // Round2 as the spec defines it — arithmetic shift, so ties round toward +infinity. Matching the
    // CPU exactly is what makes tiled and full-frame renders identical rather than merely similar.
    static inline int round2(int value, int shift) {
        if (shift <= 0) { return value; }
        return (value + (1 << (shift - 1))) >> shift;
    }

    static inline int clampGrain(int v) { return clamp(v, kGrainMin, kGrainMax); }

    static inline int sampleTemplate(device const int *tmpl, int2 offset, int x, int y) {
        int sx = kSampleOrigin + offset.x * 2 + x;
        int sy = kSampleOrigin + offset.y * 2 + y;
        return tmpl[sy * kTemplateWidth + sx];
    }

    kernel void fg_apply(texture2d<float, access::read> src [[texture(0)]],
                         texture2d<float, access::write> dst [[texture(1)]],
                         device const int *tmpl [[buffer(0)]],
                         device const int *scaling [[buffer(1)]],
                         device const int2 *offsets [[buffer(2)]],
                         constant Config &cfg [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= cfg.width || gid.y >= cfg.height) { return; }

        // ── Absolute addressing. Grain belongs to a position in the PICTURE, never in the tile.
        int absX = cfg.originX + int(gid.x);
        int absY = cfg.originY + int(gid.y);
        int blockColumn = absX / kBlockSize;
        int blockRow = absY / kBlockSize;
        int x = absX - blockColumn * kBlockSize;
        int y = absY - blockRow * kBlockSize;

        int rowIndex = blockRow - cfg.firstOffsetRow;
        int2 offset = offsets[rowIndex * cfg.columns + blockColumn];
        int value = sampleTemplate(tmpl, offset, x, y);

        if (cfg.overlap != 0) {
            // Top seam — the partner is the block ABOVE, at its continuation rows 32+y. Reading the
            // neighbour's own continuation (rather than whatever was written) is what makes the field
            // independent of evaluation order, and it is why a block reads 34 samples for 32 pixels.
            if (blockRow > 0 && y < 2) {
                int2 w = kOverlapWeights[y];
                int2 above = offsets[(rowIndex - 1) * cfg.columns + blockColumn];
                int old = sampleTemplate(tmpl, above, x, kBlockSize + y);
                value = clampGrain(round2(old * w.x + value * w.y, kOverlapShift));
            }
            // Left seam — the block to the LEFT, continuation columns 32+x.
            if (blockColumn > 0 && x < 2) {
                int2 w = kOverlapWeights[x];
                int2 left = offsets[rowIndex * cfg.columns + blockColumn - 1];
                int old = sampleTemplate(tmpl, left, kBlockSize + x, y);
                // The partner carries its own top seam, or the 2×2 corner would blend against an
                // unblended neighbour.
                if (blockRow > 0 && y < 2) {
                    int2 wy = kOverlapWeights[y];
                    int2 aboveLeft = offsets[(rowIndex - 1) * cfg.columns + blockColumn - 1];
                    int older = sampleTemplate(tmpl, aboveLeft, kBlockSize + x, kBlockSize + y);
                    old = clampGrain(round2(older * wy.x + old * wy.y, kOverlapShift));
                }
                value = clampGrain(round2(old * w.x + value * w.y, kOverlapShift));
            }
        }

        float4 colour = src.read(gid);
        // Amplitude is looked up by the SOURCE brightness. Feeding the curve the already-grained value
        // would make the effect depend on its own output.
        float luma = cfg.isLuma != 0 ? colour.r : dot(colour.rgb, kLumaWeights);
        int level = clamp(int(round(luma * 255.0f)), 0, 255);
        int scaled = round2(scaling[level] * value, cfg.scalingShift);

        if (cfg.isLuma != 0) {
            float out = clamp(luma * 255.0f + float(scaled), 0.0f, 255.0f) / 255.0f;
            dst.write(float4(out, 0, 0, 1), gid);
        } else {
            // Equal delta on R, G and B leaves the colour differences untouched — grain is a luma
            // effect, and chroma noise reads as digital rather than photographic.
            float delta = (clamp(luma * 255.0f + float(scaled), 0.0f, 255.0f) - luma * 255.0f) / 255.0f;
            dst.write(float4(clamp(colour.rgb + delta, 0.0f, 1.0f), colour.a), gid);
        }
    }
    """
}
