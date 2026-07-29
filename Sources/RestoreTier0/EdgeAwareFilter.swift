//
// EdgeAwareFilter.swift — RestoreTier0
//
// N1's base/detail separator. See `mlxengine-todo/GAP-PROGRAM.md` §N1.
//
// 🔑 **Why this is a protocol with a CPU implementation rather than a direct `MPSImageGuidedFilter`
// call, which is what the plan prescribes.** Two reasons, and the first is structural:
//
//   1. **Category A forbids a Metal requirement.** `RestoreTier0` is net-clean, macOS-14, no Metal —
//      that is the guarantee that lets a host consume the planner without being forced into a GPU
//      toolchain. A hard MPS dependency would quietly revoke it for every consumer of this package.
//   2. **V7 prescribed exactly this hedge anyway.** The spike that unblocked N1 found
//      `MPSImageGuidedFilter` is *not* deprecated, but flagged one real risk: in macOS 27 only
//      `MPSNDArray` gained `encodeWithMTL4CommandEncoder`, so a Metal 4 pipeline needs a separate
//      classic queue plus `MTLEvent` sync. Its recommended mitigation was *"isolate behind an internal
//      `EdgeAwareFilter` protocol — free today, one-file swap later."* This is that protocol.
//
// The precedent is already in the tree: `media-bridge`'s SSIMULACRA2 keeps the validated CPU path and
// injects a Metal backend for the one hot stage. Same shape here — and the CPU implementation stays the
// correctness oracle for any accelerated backend that lands later.
//
// ⚠️ **The filter is the guided filter, not a bilateral one, and that is not a convenience.**
// `MPSImageBilateral` does not exist; more importantly He/Sun/Tang (TPAMI 2013) is **free of the
// gradient-reversal artifacts** that ruin a bilateral detail layer *exactly at the edges you are
// sharpening* — which is the failure this whole row is trying to avoid.
//

import Foundation

/// A base/detail separator: given a plane, return the smoothed **base** layer. Detail is `input − base`.
///
/// `epsilon` is in **intensity² units**, which is the entire noise-awareness mechanism: set it to
/// `(k·σ)²` and any local variation below the noise floor is treated as flat, lands in *base*, and
/// never enters *detail*. Not an approximation of noise-awareness — the exact thing, in one parameter.
public protocol EdgeAwareFilter: Sendable {
    func base(of plane: [Float], width: Int, height: Int, radius: Int, epsilon: Float) -> [Float]
}

/// Guided filter (He/Sun/Tang), self-guided, on the CPU.
///
/// Self-guided means `I == p`: the plane guides its own smoothing, which is what makes it an
/// edge-preserving base/detail split rather than a joint filter.
public struct GuidedFilter: EdgeAwareFilter {

    public init() {}

    public func base(of plane: [Float], width: Int, height: Int,
                     radius: Int, epsilon: Float) -> [Float] {
        precondition(plane.count == width * height)
        guard radius > 0, width > 0, height > 0 else { return plane }

        let meanI = BoxFilter.apply(plane, width: width, height: height, radius: radius)
        let squares = zip(plane, plane).map(*)
        let meanII = BoxFilter.apply(squares, width: width, height: height, radius: radius)

        var a = [Float](repeating: 0, count: plane.count)
        var b = [Float](repeating: 0, count: plane.count)
        for i in 0..<plane.count {
            let variance = max(0, meanII[i] - meanI[i] * meanI[i])
            // ⚠️ `variance / (variance + ε)` is genuinely 0/0 on a flat region of a noiseless source,
            // and both halves of that are ordinary inputs — a blank frame, a blown sky, a solid
            // background — not edge cases. The mathematical limit is `a = 0` (no structure ⇒ pure
            // smoothing), so take it explicitly rather than shipping a NaN that spreads through the
            // two box filters below and poisons the whole plane.
            let denominator = variance + epsilon
            a[i] = denominator > 0 ? variance / denominator : 0
            b[i] = (1 - a[i]) * meanI[i]
        }

        let meanA = BoxFilter.apply(a, width: width, height: height, radius: radius)
        let meanB = BoxFilter.apply(b, width: width, height: height, radius: radius)

        var out = [Float](repeating: 0, count: plane.count)
        for i in 0..<plane.count { out[i] = meanA[i] * plane[i] + meanB[i] }
        return out
    }

    /// Local variance at `radius` — the guided filter already computes this internally, and the
    /// Polesel gate in `NoiseAwareSharpener` needs it, so it is exposed rather than recomputed.
    public static func localVariance(_ plane: [Float], width: Int, height: Int,
                                     radius: Int) -> [Float] {
        let mean = BoxFilter.apply(plane, width: width, height: height, radius: radius)
        let meanSquares = BoxFilter.apply(zip(plane, plane).map(*), width: width, height: height,
                                          radius: radius)
        return (0..<plane.count).map { max(0, meanSquares[$0] - mean[$0] * mean[$0]) }
    }
}

/// Separable box filter with edge clamp, O(1) per pixel via a rolling sum.
public enum BoxFilter {

    /// 🚨 **The accumulator is `Double`, and that is load-bearing, not fastidiousness.** The plan warns
    /// against `MPSImageIntegral` for local variance at 50 MP: an integral image's accumulator reaches
    /// ~5×10⁷ against fp32's 24-bit mantissa (~1.7×10⁷), and the catastrophic cancellation when you
    /// subtract two large partial sums produces **visible banding**.
    ///
    /// A rolling box sum sidesteps most of that by construction — it only ever holds one window, not a
    /// running total over the whole image — but the running value still drifts across a long row as
    /// terms are added and subtracted. A `Double` scalar accumulator removes the drift entirely at a
    /// cost of nothing: it is one scalar per row, not a second image-sized buffer, so the memory
    /// argument that rules out a Double *plane* does not apply here.
    public static func apply(_ src: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        precondition(src.count == width * height)
        guard radius > 0 else { return src }

        var horizontal = [Float](repeating: 0, count: width * height)
        // Edge clamping replicates border samples, so every window holds exactly 2r+1 terms —
        // the divisor is constant and no per-pixel count is needed.
        let window = Double(2 * radius + 1)
        for y in 0..<height {
            let row = y * width
            var sum = 0.0
            // Prime the window at x = 0, with clamped edges.
            for k in -radius...radius { sum += Double(src[row + clamp(k, width)]) }
            for x in 0..<width {
                horizontal[row + x] = Float(sum / window)
                sum += Double(src[row + clamp(x + radius + 1, width)])
                sum -= Double(src[row + clamp(x - radius, width)])
            }
        }

        var out = [Float](repeating: 0, count: width * height)
        for x in 0..<width {
            var sum = 0.0
            for k in -radius...radius { sum += Double(horizontal[clamp(k, height) * width + x]) }
            for y in 0..<height {
                out[y * width + x] = Float(sum / window)
                sum += Double(horizontal[clamp(y + radius + 1, height) * width + x])
                sum -= Double(horizontal[clamp(y - radius, height) * width + x])
            }
        }
        return out
    }

    @inline(__always)
    private static func clamp(_ v: Int, _ extent: Int) -> Int { min(max(v, 0), extent - 1) }
}
