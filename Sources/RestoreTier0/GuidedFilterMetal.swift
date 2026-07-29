//
// GuidedFilterMetal.swift — RestoreTier0
//
// GPU backend for N1's base/detail separator, and a **drop-in `EdgeAwareFilter`** — so
// `NoiseAwareSharpener(filter: GuidedFilterMetal.shared ?? GuidedFilter())` is the whole integration.
//
// 🔑 **Why a Metal file is fine in a category-A package, since this looked like a contradiction.**
// The rule is *"no Metal **requirement**"* — the package must build and run without a GPU, so a host
// that wants the planner is never forced into a GPU toolchain or a 26.x floor. It is not a ban on
// Metal. The in-tree precedent is `media-bridge`, which is category A **and** ships
// `SSIMULACRA2Metal`. Three properties keep that true, and this file has all three:
//
//   1. **`init?()` returns nil when there is no device** — every caller has a CPU path.
//   2. **The shader is compiled at runtime from a source string** (`makeLibrary(source:)`), so there
//      is no `.metal` file, no metallib to package, and no Metal build-system dependency. That also
//      sidesteps the metallib trap this codebase has already paid for once.
//   3. **The CPU implementation stays the oracle.** Parity is tested, not assumed.
//
// ⚠️ **Do not read this as the GPU-residency story — it is one stage of it.** This accelerates the
// filter; the plane in and out is still `[Float]`, so a caller holding an `MTLTexture` pays a round
// trip. Closing that is the `CVPixelBuffer`/`IOSurface` currency work (`CLAUDE.md`: *"keep pixels
// GPU-resident… not `CGImage`, which forces a CPU copy"*), and it is a separate, larger change that
// has to move the whole sharpen chain — coring, gate, Sobel, clamp — onto the GPU too. Staged the same
// way `SSIMULACRA2Metal` staged its own port: the dominant, parity-critical stage first.
//
// 🔑 **Accuracy note that inverts the usual expectation.** The GPU box filter sums its window fresh
// per pixel; the CPU one rolls a running sum along the row. So the **GPU path is the more accurate of
// the two** — the CPU's `Double` accumulator exists precisely to keep that difference negligible. Do
// not treat a small CPU/GPU delta as a GPU bug.
//

import Foundation
import Metal

public final class GuidedFilterMetal: EdgeAwareFilter, @unchecked Sendable {

    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let squarePipe: MTLComputePipelineState
    private let boxHPipe: MTLComputePipelineState
    private let boxVPipe: MTLComputePipelineState
    private let abPipe: MTLComputePipelineState
    private let reconstructPipe: MTLComputePipelineState

    public init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = try? device.makeLibrary(source: Self.kernelSource, options: nil),
              let square = library.makeFunction(name: "gf_square"),
              let boxH = library.makeFunction(name: "gf_box_h"),
              let boxV = library.makeFunction(name: "gf_box_v"),
              let ab = library.makeFunction(name: "gf_ab"),
              let reconstruct = library.makeFunction(name: "gf_reconstruct"),
              let squarePipe = try? device.makeComputePipelineState(function: square),
              let boxHPipe = try? device.makeComputePipelineState(function: boxH),
              let boxVPipe = try? device.makeComputePipelineState(function: boxV),
              let abPipe = try? device.makeComputePipelineState(function: ab),
              let reconstructPipe = try? device.makeComputePipelineState(function: reconstruct)
        else { return nil }

        self.device = device
        self.queue = queue
        self.squarePipe = squarePipe
        self.boxHPipe = boxHPipe
        self.boxVPipe = boxVPipe
        self.abPipe = abPipe
        self.reconstructPipe = reconstructPipe
    }

    /// Shared instance — the shader compiles once. `nil` when no Metal device is available, which is
    /// the caller's signal to use `GuidedFilter()`.
    public static let shared = GuidedFilterMetal()

    /// Says *where* GPU setup failed — a missing device (a host can inject one) reads differently from
    /// a `makeLibrary(source:)` failure (needs a precompiled metallib instead). Same reasoning as
    /// `SSIMULACRA2Metal.diagnostics()`.
    public static func diagnostics() -> String {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return "FAIL: no Metal device (MTLCreateSystemDefaultDevice == nil)"
        }
        guard device.makeCommandQueue() != nil else {
            return "FAIL: \(device.name): makeCommandQueue == nil"
        }
        do {
            let library = try device.makeLibrary(source: kernelSource, options: nil)
            let names = ["gf_square", "gf_box_h", "gf_box_v", "gf_ab", "gf_reconstruct"]
            for name in names where library.makeFunction(name: name) == nil {
                return "FAIL: \(device.name): missing kernel \(name)"
            }
            return "OK \(device.name): runtime shader compile + all pipelines available"
        } catch {
            return "FAIL: \(device.name): makeLibrary(source:) — \(error.localizedDescription)"
        }
    }

    // MARK: - EdgeAwareFilter

    public func base(of plane: [Float], width: Int, height: Int,
                     radius: Int, epsilon: Float) -> [Float] {
        precondition(plane.count == width * height)
        guard radius > 0, width > 0, height > 0 else { return plane }

        let count = width * height
        let bytes = count * MemoryLayout<Float>.stride
        // The CPU path is not merely a fallback for "no device" — an allocation failure mid-frame must
        // degrade rather than trap, because this runs per tile in an interactive loop.
        guard let source = device.makeBuffer(bytes: plane, length: bytes, options: .storageModeShared),
              let scratchA = device.makeBuffer(length: bytes, options: .storageModeShared),
              let scratchB = device.makeBuffer(length: bytes, options: .storageModeShared),
              let meanI = device.makeBuffer(length: bytes, options: .storageModeShared),
              let meanII = device.makeBuffer(length: bytes, options: .storageModeShared),
              let aBuffer = device.makeBuffer(length: bytes, options: .storageModeShared),
              let bBuffer = device.makeBuffer(length: bytes, options: .storageModeShared),
              let output = device.makeBuffer(length: bytes, options: .storageModeShared),
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder()
        else {
            return GuidedFilter().base(of: plane, width: width, height: height,
                                       radius: radius, epsilon: epsilon)
        }

        var dims = SIMD2<UInt32>(UInt32(width), UInt32(height))
        var elementCount = UInt32(count)
        var radiusValue = Int32(radius)
        var epsilonValue = epsilon

        let grid2D = MTLSize(width: width, height: height, depth: 1)
        let group2D = MTLSize(width: 16, height: 16, depth: 1)
        let grid1D = MTLSize(width: count, height: 1, depth: 1)
        let group1D = MTLSize(width: 64, height: 1, depth: 1)

        /// One separable box pass: `input → scratch → destination`.
        func box(_ input: MTLBuffer, into destination: MTLBuffer) {
            encoder.setComputePipelineState(boxHPipe)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(scratchA, offset: 0, index: 1)
            encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            encoder.setBytes(&radiusValue, length: MemoryLayout<Int32>.size, index: 3)
            encoder.dispatchThreads(grid2D, threadsPerThreadgroup: group2D)

            encoder.setComputePipelineState(boxVPipe)
            encoder.setBuffer(scratchA, offset: 0, index: 0)
            encoder.setBuffer(destination, offset: 0, index: 1)
            encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            encoder.setBytes(&radiusValue, length: MemoryLayout<Int32>.size, index: 3)
            encoder.dispatchThreads(grid2D, threadsPerThreadgroup: group2D)
        }

        // mean(I)
        box(source, into: meanI)

        // mean(I²)
        encoder.setComputePipelineState(squarePipe)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(scratchB, offset: 0, index: 1)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreads(grid1D, threadsPerThreadgroup: group1D)
        box(scratchB, into: meanII)

        // a, b
        encoder.setComputePipelineState(abPipe)
        encoder.setBuffer(meanI, offset: 0, index: 0)
        encoder.setBuffer(meanII, offset: 0, index: 1)
        encoder.setBuffer(aBuffer, offset: 0, index: 2)
        encoder.setBuffer(bBuffer, offset: 0, index: 3)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&epsilonValue, length: MemoryLayout<Float>.size, index: 5)
        encoder.dispatchThreads(grid1D, threadsPerThreadgroup: group1D)

        // mean(a), mean(b) — reusing meanI/meanII, which are dead from here.
        box(aBuffer, into: meanI)
        box(bBuffer, into: meanII)

        // q = mean(a)·I + mean(b)
        encoder.setComputePipelineState(reconstructPipe)
        encoder.setBuffer(meanI, offset: 0, index: 0)
        encoder.setBuffer(meanII, offset: 0, index: 1)
        encoder.setBuffer(source, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.dispatchThreads(grid1D, threadsPerThreadgroup: group1D)

        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()

        let pointer = output.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    // MARK: - Kernels

    /// Compiled at runtime, so there is no `.metal` file and no metallib to package — see the header.
    static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void gf_square(device const float *src [[buffer(0)]],
                          device float *dst [[buffer(1)]],
                          constant uint &count [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
        if (gid >= count) { return; }
        float v = src[gid];
        dst[gid] = v * v;
    }

    // Edge-clamped box, summed fresh per pixel — every window holds exactly 2r+1 terms, which is why
    // the divisor is constant and the frame border does not darken.
    kernel void gf_box_h(device const float *src [[buffer(0)]],
                         device float *dst [[buffer(1)]],
                         constant uint2 &dims [[buffer(2)]],
                         constant int &radius [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
        uint w = dims.x, h = dims.y;
        if (gid.x >= w || gid.y >= h) { return; }
        int x = int(gid.x);
        int last = int(w) - 1;
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            int sx = clamp(x + k, 0, last);
            sum += src[gid.y * w + uint(sx)];
        }
        dst[gid.y * w + gid.x] = sum / float(2 * radius + 1);
    }

    kernel void gf_box_v(device const float *src [[buffer(0)]],
                         device float *dst [[buffer(1)]],
                         constant uint2 &dims [[buffer(2)]],
                         constant int &radius [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
        uint w = dims.x, h = dims.y;
        if (gid.x >= w || gid.y >= h) { return; }
        int y = int(gid.y);
        int last = int(h) - 1;
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            int sy = clamp(y + k, 0, last);
            sum += src[uint(sy) * w + gid.x];
        }
        dst[gid.y * w + gid.x] = sum / float(2 * radius + 1);
    }

    // a = var / (var + eps), guarded at 0/0 — a flat region of a noiseless source is an ordinary
    // input (a blank frame, a blown sky), and the limit there is a = 0.
    kernel void gf_ab(device const float *meanI [[buffer(0)]],
                      device const float *meanII [[buffer(1)]],
                      device float *a [[buffer(2)]],
                      device float *b [[buffer(3)]],
                      constant uint &count [[buffer(4)]],
                      constant float &epsilon [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
        if (gid >= count) { return; }
        float mi = meanI[gid];
        float variance = max(0.0f, meanII[gid] - mi * mi);
        float denominator = variance + epsilon;
        float av = denominator > 0.0f ? variance / denominator : 0.0f;
        a[gid] = av;
        b[gid] = (1.0f - av) * mi;
    }

    kernel void gf_reconstruct(device const float *meanA [[buffer(0)]],
                               device const float *meanB [[buffer(1)]],
                               device const float *src [[buffer(2)]],
                               device float *dst [[buffer(3)]],
                               constant uint &count [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
        if (gid >= count) { return; }
        dst[gid] = meanA[gid] * src[gid] + meanB[gid];
    }
    """
}
