//
// MetalSharpenPipeline.swift — RestoreTier0
//
// N1's whole chain on the GPU, entered and left as a `CVPixelBuffer`. **This is the surface an app
// should consume**; the `[Float]` API on `NoiseAwareSharpener` is the reference implementation and the
// parity oracle behind it.
//
// 🚨 **Why the whole chain and not just the filter.** Accelerating one stage while the rest stays on
// the CPU produces `texture → plane → GPU → plane → … → texture`, which is **slower than never
// touching the GPU at all** — two transfers per stage, and the transfer dominates a per-pixel kernel.
// So every stage lives here: luma ingest, the noise estimate, three guided-filter bands, coring, the
// Polesel gate, Sobel roll-off, accumulation, the asymmetric envelope clamp, and write-back. The
// pixel buffer is touched twice — once in, once out — and never in between.
//
// Signature note: `process(_:) -> CVPixelBuffer` deliberately matches `ImageBridge.FrameProcessor`
// **without importing media-bridge**. Taking that dependency to inherit one method would pull the
// whole media foundation into a Tier-0 restore package. A host that wants the conformance writes
// `extension MetalSharpenPipeline: FrameProcessor {}` — one line, and it costs this package nothing.
//
// ⚠️ **One unavoidable sync point.** σ sets ε, the coring knee *and* the gate, so nothing downstream
// can be encoded until it is known — and it is a reduction. Hence two committed passes rather than
// one. A caller that already knows σ (measure once, then drag a slider) sets `options.noiseSigma` and
// the first pass collapses to the ingest alone.
//
// **Composes with `MetalFilmGrain`** on the same `CVPixelBuffer` currency, so sharpen → grain chains
// with no readback. Both take and return IOSurface-backed buffers, which the texture cache maps
// zero-copy. *(This note previously said grain was still plane-only; that stopped being true in
// v0.5.0.)*
//

import CoreVideo
import ForgePixelBridge
import Foundation
import Metal

public final class MetalSharpenPipeline: @unchecked Sendable {

    public struct Failure: Error, Equatable {
        public let reason: String
    }

    /// What the pass measured. Mirrors `NoiseAwareSharpener.Report`'s receipt fields. The two
    /// statistics that would need a GPU→CPU readback — mean change and clamp fraction — are omitted
    /// rather than paid for on every frame; `NoiseAwareSharpener` still reports them when wanted.
    public struct Report: Sendable, Equatable {
        public let noiseSigma: Float
        public let sigmaWasSupplied: Bool
        public let appliedBandDiameters: [Int]
        public let width: Int
        public let height: Int
        public let layout: PixelBufferBridge.Layout
    }

    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let cache: CVMetalTextureCache
    private let pipelines: [String: MTLComputePipelineState]

    public var options: NoiseAwareSharpener.Options

    public init?(options: NoiseAwareSharpener.Options = .default) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = try? device.makeLibrary(source: Self.kernelSource, options: nil),
              let cache = try? PixelBufferBridge.makeTextureCache(device: device)
        else { return nil }

        var built: [String: MTLComputePipelineState] = [:]
        for name in Self.kernelNames {
            guard let function = library.makeFunction(name: name),
                  let pipeline = try? device.makeComputePipelineState(function: function)
            else { return nil }
            built[name] = pipeline
        }

        self.device = device
        self.queue = queue
        self.cache = cache
        self.pipelines = built
        self.options = options
    }

    static let kernelNames = [
        "sp_ingest_bgra", "sp_ingest_luma", "sp_write_bgra", "sp_write_luma",
        "sp_box_h", "sp_box_v", "sp_square", "sp_copy", "sp_zero",
        "sp_ab", "sp_reconstruct", "sp_sobel", "sp_band",
        "sp_immerkaer_rows", "sp_combine_clamp",
    ]

    /// Names *which* stage of setup failed — three problems with three different fixes.
    public static func diagnostics() -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "FAIL: no Metal device" }
        guard device.makeCommandQueue() != nil else { return "FAIL: \(device.name): no command queue" }
        do {
            let library = try device.makeLibrary(source: kernelSource, options: nil)
            for name in kernelNames where library.makeFunction(name: name) == nil {
                return "FAIL: \(device.name): missing kernel \(name)"
            }
            return "OK \(device.name): \(kernelNames.count) kernels available"
        } catch {
            return "FAIL: \(device.name): makeLibrary(source:) — \(error.localizedDescription)"
        }
    }

    // MARK: - Entry points

    /// Sharpen into a fresh buffer. Matches `FrameProcessor.process` — on failure it returns the input
    /// **unchanged** rather than trapping, because a per-frame path must degrade, not die.
    public func process(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer {
        (try? sharpen(pixelBuffer)) ?? pixelBuffer
    }

    public func sharpen(_ pixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer {
        guard let destination = PixelBufferBridge.makeCompatibleBuffer(like: pixelBuffer) else {
            throw Failure(reason: "could not allocate a destination buffer")
        }
        _ = try run(source: pixelBuffer, destination: destination)
        return destination
    }

    /// Sharpen `source` into a `destination` the caller owns — the form a render loop wants, with no
    /// per-frame allocation.
    @discardableResult
    public func run(source: CVPixelBuffer, destination: CVPixelBuffer) throws -> Report {
        let layout = try PixelBufferBridge.Layout.detect(source)
        guard try PixelBufferBridge.Layout.detect(destination) == layout else {
            throw Failure(reason: "source and destination layouts differ")
        }
        let format = PixelBufferBridge.pixelFormat(for: layout, pixelBuffer: source)

        let (sourceWrapper, sourceTexture) = try PixelBufferBridge.texture(
            from: source, plane: 0, pixelFormat: format, cache: cache)
        let (destinationWrapper, destinationTexture) = try PixelBufferBridge.texture(
            from: destination, plane: 0, pixelFormat: format, cache: cache)
        // The MTLTextures alias these wrappers; releasing one while its texture is in flight is a
        // use-after-free that usually *appears* to work.
        defer { withExtendedLifetime((sourceWrapper, destinationWrapper)) {} }

        let width = sourceTexture.width, height = sourceTexture.height
        guard width >= 3, height >= 3 else { throw Failure(reason: "picture too small") }
        guard destinationTexture.width == width, destinationTexture.height == height else {
            throw Failure(reason: "source and destination dimensions differ")
        }
        let count = width * height

        // luma · sobel · accumulated · current · base · variance · four scratch + the row sums.
        guard let luma = buffer(count), let sobel = buffer(count),
              let accumulated = buffer(count), let current = buffer(count),
              let base = buffer(count), let variance = buffer(count),
              let t0 = buffer(count), let t1 = buffer(count),
              let t2 = buffer(count), let t3 = buffer(count),
              let rowSums = device.makeBuffer(length: max(1, height) * MemoryLayout<Float>.stride,
                                              options: .storageModeShared)
        else { throw Failure(reason: "buffer allocation failed") }

        let dims = SIMD2<UInt32>(UInt32(width), UInt32(height))
        let elements = UInt32(count)

        // ─── Pass 1 — ingest, and the noise estimate when it is not supplied ──────────────────
        guard let ingest = queue.makeCommandBuffer(),
              let ingestEncoder = ingest.makeComputeCommandEncoder() else {
            throw Failure(reason: "could not encode the ingest pass")
        }
        encode(ingestEncoder, layout == .packedBGRA ? "sp_ingest_bgra" : "sp_ingest_luma",
               textures: [sourceTexture], buffers: [luma], constants: [Self.raw(dims)],
               grid: MTLSize(width: width, height: height, depth: 1))
        let sigmaWasSupplied = options.noiseSigma != nil
        if !sigmaWasSupplied {
            encode(ingestEncoder, "sp_immerkaer_rows", textures: [], buffers: [luma, rowSums],
                   constants: [Self.raw(dims)], grid: MTLSize(width: height, height: 1, depth: 1))
        }
        ingestEncoder.endEncoding()
        ingest.commit()
        ingest.waitUntilCompleted()
        if let error = ingest.error { throw Failure(reason: "GPU ingest: \(error.localizedDescription)") }

        let sigma: Float
        if let supplied = options.noiseSigma {
            sigma = supplied
        } else {
            let rows = rowSums.contents().bindMemory(to: Float.self, capacity: height)
            var total = 0.0
            for y in 0..<height { total += Double(rows[y]) }
            let samples = Double(max(1, (width - 2) * (height - 2)))
            sigma = Float(total / samples * (Double.pi / 2).squareRoot() / 6.0)
        }

        let noiseFloor = options.noiseFloorK * Double(sigma)
        let epsilon = Float(pow(max(options.detailScale, noiseFloor), 2))
        let gateKnee = Float(pow(noiseFloor, 2))
        let coring = Float(options.coringC * Double(sigma))
        let maxGain = Float(options.maxGain)
        let edgeKnee = Float(options.edgeKnee)
        let undershoot = Float(options.undershootAllowance)
        let overshoot = Float(options.overshootAllowance)
        let floorValue = Float(noiseFloor)

        // ─── Pass 2 — the entire chain, one encoder, no readback ──────────────────────────────
        guard let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else {
            throw Failure(reason: "could not encode the main pass")
        }
        let grid2D = MTLSize(width: width, height: height, depth: 1)
        let grid1D = MTLSize(width: count, height: 1, depth: 1)

        encode(encoder, "sp_sobel", textures: [], buffers: [luma, sobel],
               constants: [Self.raw(dims)], grid: grid2D)
        encode(encoder, "sp_copy", textures: [], buffers: [luma, current],
               constants: [Self.raw(elements)], grid: grid1D)
        encode(encoder, "sp_zero", textures: [], buffers: [accumulated],
               constants: [Self.raw(elements)], grid: grid1D)

        var appliedDiameters: [Int] = []
        for (index, diameter) in options.bandDiameters.enumerated() {
            // A band wider than the picture is a DC offset, not a band — see `NoiseAwareSharpener`.
            guard diameter <= min(width, height) else { continue }
            appliedDiameters.append(diameter)
            let radius = Int32(max(1, diameter / 2))
            let gain = Float(options.amount *
                             (index < options.bandGains.count ? options.bandGains[index] : 0))

            box(current, into: t0, scratch: t3, radius: radius, dims: dims, encoder: encoder)  // meanI
            encode(encoder, "sp_square", textures: [], buffers: [current, t1],
                   constants: [Self.raw(elements)], grid: grid1D)
            box(t1, into: t2, scratch: t3, radius: radius, dims: dims, encoder: encoder)       // meanII
            encode(encoder, "sp_ab", textures: [], buffers: [t0, t2, t1, t2, variance],
                   constants: [Self.raw(elements), Self.raw(epsilon)], grid: grid1D)                          // a→t1, b→t2
            box(t1, into: t0, scratch: t3, radius: radius, dims: dims, encoder: encoder)       // mean_a
            box(t2, into: t1, scratch: t3, radius: radius, dims: dims, encoder: encoder)       // mean_b
            encode(encoder, "sp_reconstruct", textures: [], buffers: [t0, t1, current, base],
                   constants: [Self.raw(elements)], grid: grid1D)

            encode(encoder, "sp_band", textures: [],
                   buffers: [current, base, variance, sobel, accumulated],
                   constants: [Self.raw(elements), Self.raw(coring), Self.raw(gateKnee),
                               Self.raw(maxGain), Self.raw(edgeKnee), Self.raw(gain)],
                   grid: grid1D)

            encode(encoder, "sp_copy", textures: [], buffers: [base, current],
                   constants: [Self.raw(elements)], grid: grid1D)
        }

        // Combine onto the ORIGINAL — the decomposition telescopes — then clamp, then write back.
        encode(encoder, "sp_combine_clamp", textures: [], buffers: [luma, accumulated, t0],
               constants: [Self.raw(dims), Self.raw(undershoot), Self.raw(overshoot), Self.raw(floorValue)],
               grid: grid2D)
        encode(encoder, layout == .packedBGRA ? "sp_write_bgra" : "sp_write_luma",
               textures: [sourceTexture, destinationTexture], buffers: [luma, t0],
               constants: [Self.raw(dims)], grid: grid2D)

        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw Failure(reason: "GPU: \(error.localizedDescription)") }

        return Report(noiseSigma: sigma, sigmaWasSupplied: sigmaWasSupplied,
                      appliedBandDiameters: appliedDiameters,
                      width: width, height: height, layout: layout)
    }

    // MARK: - Encoding helpers

    private func buffer(_ count: Int) -> MTLBuffer? {
        device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModePrivate)
    }

    /// Flatten a value to its bytes for `setBytes`. Taking `&value` inline is not expressible in an
    /// array literal, and these payloads are 4–16 bytes called a few dozen times a frame, so the
    /// copy is free relative to a single dispatch.
    static func raw<T>(_ value: T) -> [UInt8] { withUnsafeBytes(of: value) { Array($0) } }

    private func encode(_ encoder: MTLComputeCommandEncoder, _ name: String,
                        textures: [MTLTexture], buffers: [MTLBuffer],
                        constants: [[UInt8]], grid: MTLSize) {
        guard let pipeline = pipelines[name] else { return }
        encoder.setComputePipelineState(pipeline)
        for (i, t) in textures.enumerated() { encoder.setTexture(t, index: i) }
        for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        for (i, c) in constants.enumerated() {
            encoder.setBytes(c, length: c.count, index: buffers.count + i)
        }
        let group = grid.height > 1
            ? MTLSize(width: 16, height: 16, depth: 1)
            : MTLSize(width: 64, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: group)
    }

    private func box(_ input: MTLBuffer, into destination: MTLBuffer, scratch: MTLBuffer,
                     radius: Int32, dims: SIMD2<UInt32>,
                     encoder: MTLComputeCommandEncoder) {
        let grid = MTLSize(width: Int(dims.x), height: Int(dims.y), depth: 1)
        let constants = [Self.raw(dims), Self.raw(radius)]
        encode(encoder, "sp_box_h", textures: [], buffers: [input, scratch],
               constants: constants, grid: grid)
        encode(encoder, "sp_box_v", textures: [], buffers: [scratch, destination],
               constants: constants, grid: grid)
    }

    // MARK: - Kernels

    /// Compiled at runtime — no `.metal` file, no metallib to package, no Metal build dependency.
    static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    constant float3 kLumaWeights = float3(0.2126f, 0.7152f, 0.0722f);

    // ── Ingest ───────────────────────────────────────────────────────────────────────────────
    // Rec.709 luma from GAMMA-ENCODED sRGB. The texture must NOT be an _srgb format, or Metal
    // linearizes on read and every σ-derived constant in this package is calibrated for the wrong
    // domain — see PixelBufferBridge.
    kernel void sp_ingest_bgra(texture2d<float, access::read> src [[texture(0)]],
                               device float *luma [[buffer(0)]],
                               constant uint2 &dims [[buffer(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dims.x || gid.y >= dims.y) { return; }
        float4 c = src.read(gid);
        luma[gid.y * dims.x + gid.x] = dot(c.rgb, kLumaWeights);
    }

    kernel void sp_ingest_luma(texture2d<float, access::read> src [[texture(0)]],
                               device float *luma [[buffer(0)]],
                               constant uint2 &dims [[buffer(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dims.x || gid.y >= dims.y) { return; }
        luma[gid.y * dims.x + gid.x] = src.read(gid).r;
    }

    // ── Write-back ───────────────────────────────────────────────────────────────────────────
    // The luma delta is added equally to R, G and B, which leaves the colour differences (R−Y, B−Y)
    // unchanged — chroma preserved by construction, not by care.
    kernel void sp_write_bgra(texture2d<float, access::read> src [[texture(0)]],
                              texture2d<float, access::write> dst [[texture(1)]],
                              device const float *before [[buffer(0)]],
                              device const float *after [[buffer(1)]],
                              constant uint2 &dims [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dims.x || gid.y >= dims.y) { return; }
        uint i = gid.y * dims.x + gid.x;
        float4 c = src.read(gid);
        float delta = after[i] - before[i];
        dst.write(float4(clamp(c.rgb + delta, 0.0f, 1.0f), c.a), gid);
    }

    kernel void sp_write_luma(texture2d<float, access::read> src [[texture(0)]],
                              texture2d<float, access::write> dst [[texture(1)]],
                              device const float *before [[buffer(0)]],
                              device const float *after [[buffer(1)]],
                              constant uint2 &dims [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dims.x || gid.y >= dims.y) { return; }
        uint i = gid.y * dims.x + gid.x;
        dst.write(float4(clamp(after[i], 0.0f, 1.0f), 0, 0, 1), gid);
    }

    // ── Box filter ───────────────────────────────────────────────────────────────────────────
    // Edge-clamped, summed fresh per pixel: every window holds exactly 2r+1 terms, so the divisor is
    // constant and the frame border does not darken.
    kernel void sp_box_h(device const float *src [[buffer(0)]],
                         device float *dst [[buffer(1)]],
                         constant uint2 &dims [[buffer(2)]],
                         constant int &radius [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dims.x || gid.y >= dims.y) { return; }
        int last = int(dims.x) - 1;
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            sum += src[gid.y * dims.x + uint(clamp(int(gid.x) + k, 0, last))];
        }
        dst[gid.y * dims.x + gid.x] = sum / float(2 * radius + 1);
    }

    kernel void sp_box_v(device const float *src [[buffer(0)]],
                         device float *dst [[buffer(1)]],
                         constant uint2 &dims [[buffer(2)]],
                         constant int &radius [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dims.x || gid.y >= dims.y) { return; }
        int last = int(dims.y) - 1;
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            sum += src[uint(clamp(int(gid.y) + k, 0, last)) * dims.x + gid.x];
        }
        dst[gid.y * dims.x + gid.x] = sum / float(2 * radius + 1);
    }

    // ── Small ops ────────────────────────────────────────────────────────────────────────────
    kernel void sp_square(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
                          constant uint &count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
        if (gid >= count) { return; }
        float v = src[gid];
        dst[gid] = v * v;
    }

    kernel void sp_copy(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
                        constant uint &count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
        if (gid >= count) { return; }
        dst[gid] = src[gid];
    }

    kernel void sp_zero(device float *dst [[buffer(0)]],
                        constant uint &count [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
        if (gid >= count) { return; }
        dst[gid] = 0.0f;
    }

    // a = var/(var+eps), b = (1-a)*meanI, and the local variance the Polesel gate needs.
    // The 0/0 guard is not defensive padding: a flat region of a noiseless source is an ordinary
    // input — a blank frame, a blown sky — and the limit there is a = 0.
    kernel void sp_ab(device const float *meanI [[buffer(0)]],
                      device const float *meanII [[buffer(1)]],
                      device float *a [[buffer(2)]],
                      device float *b [[buffer(3)]],
                      device float *variance [[buffer(4)]],
                      constant uint &count [[buffer(5)]],
                      constant float &epsilon [[buffer(6)]],
                      uint gid [[thread_position_in_grid]]) {
        if (gid >= count) { return; }
        float mi = meanI[gid];
        float v = max(0.0f, meanII[gid] - mi * mi);
        float denominator = v + epsilon;
        float av = denominator > 0.0f ? v / denominator : 0.0f;
        variance[gid] = v;
        a[gid] = av;
        b[gid] = (1.0f - av) * mi;
    }

    kernel void sp_reconstruct(device const float *meanA [[buffer(0)]],
                               device const float *meanB [[buffer(1)]],
                               device const float *src [[buffer(2)]],
                               device float *dst [[buffer(3)]],
                               constant uint &count [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
        if (gid >= count) { return; }
        dst[gid] = meanA[gid] * src[gid] + meanB[gid];
    }

    // ── Analysis ─────────────────────────────────────────────────────────────────────────────
    kernel void sp_sobel(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
                         constant uint2 &dims [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dims.x || gid.y >= dims.y) { return; }
        int w = int(dims.x), h = int(dims.y);
        int x = int(gid.x), y = int(gid.y);
        float s[9];
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                int sx = clamp(x + i, 0, w - 1);
                int sy = clamp(y + j, 0, h - 1);
                s[(j + 1) * 3 + (i + 1)] = src[sy * w + sx];
            }
        }
        float gx = -s[0] - 2.0f * s[3] - s[6] + s[2] + 2.0f * s[5] + s[8];
        float gy = -s[0] - 2.0f * s[1] - s[2] + s[6] + 2.0f * s[7] + s[8];
        dst[gid.y * dims.x + gid.x] = sqrt(gx * gx + gy * gy);
    }

    // The Immerkaer kernel annihilates locally-quadratic surfaces, so real image content contributes
    // almost nothing and i.i.d. noise passes with a known gain. One row per thread; the caller sums
    // the rows and applies sqrt(pi/2)/6.
    kernel void sp_immerkaer_rows(device const float *src [[buffer(0)]],
                                  device float *rowSums [[buffer(1)]],
                                  constant uint2 &dims [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
        if (gid >= dims.y) { return; }
        if (gid == 0 || gid + 1 >= dims.y) { rowSums[gid] = 0.0f; return; }
        uint w = dims.x;
        float total = 0.0f;
        for (uint x = 1; x + 1 < w; ++x) {
            uint i = gid * w + x;
            float r = src[i - w - 1] - 2.0f * src[i - w] + src[i - w + 1]
                    - 2.0f * src[i - 1] + 4.0f * src[i]     - 2.0f * src[i + 1]
                    + src[i + w - 1] - 2.0f * src[i + w] + src[i + w + 1];
            total += fabs(r);
        }
        rowSums[gid] = total;
    }

    // ── The per-band boost ───────────────────────────────────────────────────────────────────
    // coring → Polesel variance gate (keyed to the NOISE floor, not to epsilon) → Sobel roll-off.
    kernel void sp_band(device const float *current [[buffer(0)]],
                        device const float *base [[buffer(1)]],
                        device const float *variance [[buffer(2)]],
                        device const float *sobel [[buffer(3)]],
                        device float *accumulated [[buffer(4)]],
                        constant uint &count [[buffer(5)]],
                        constant float &coring [[buffer(6)]],
                        constant float &gateKnee [[buffer(7)]],
                        constant float &maxGain [[buffer(8)]],
                        constant float &edgeKnee [[buffer(9)]],
                        constant float &gain [[buffer(10)]],
                        uint gid [[thread_position_in_grid]]) {
        if (gid >= count) { return; }
        float detail = current[gid] - base[gid];
        float magnitude = fabs(detail);
        float cored = magnitude <= coring
            ? 0.0f
            : (detail < 0.0f ? -(magnitude - coring) : (magnitude - coring));

        float a2 = variance[gid];
        float variancePart = maxGain * a2 / (a2 + gateKnee + 1e-12f);
        float edge = sobel[gid] / edgeKnee;
        float edgePart = 1.0f / (1.0f + edge * edge);

        accumulated[gid] += gain * variancePart * edgePart * cored;
    }

    // ── Combine + the asymmetric halo clamp ──────────────────────────────────────────────────
    // Bright halos are far more objectionable than dark ones, so the result may undershoot the local
    // minimum considerably further than it may overshoot the local maximum. The allowance is floored
    // at the noise scale: in a flat region the local range is 0, and a purely range-proportional
    // allowance would pin the output to the source and make every pixel read as "clamped".
    kernel void sp_combine_clamp(device const float *original [[buffer(0)]],
                                 device const float *accumulated [[buffer(1)]],
                                 device float *dst [[buffer(2)]],
                                 constant uint2 &dims [[buffer(3)]],
                                 constant float &undershoot [[buffer(4)]],
                                 constant float &overshoot [[buffer(5)]],
                                 constant float &noiseFloor [[buffer(6)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dims.x || gid.y >= dims.y) { return; }
        int w = int(dims.x), h = int(dims.y);
        int x = int(gid.x), y = int(gid.y);
        float lo = 1e30f, hi = -1e30f;
        for (int j = -2; j <= 2; ++j) {
            int sy = clamp(y + j, 0, h - 1);
            for (int i = -2; i <= 2; ++i) {
                float v = original[sy * w + clamp(x + i, 0, w - 1)];
                lo = min(lo, v);
                hi = max(hi, v);
            }
        }
        uint index = gid.y * dims.x + gid.x;
        float range = max(hi - lo, noiseFloor);
        float value = original[index] + accumulated[index];
        value = clamp(value, lo - undershoot * range, hi + overshoot * range);
        dst[index] = clamp(value, 0.0f, 1.0f);
    }
    """
}
