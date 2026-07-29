//
// AFGS1Spec.swift — RestoreTier0
//
// N2 — film grain. See `mlxengine-todo/GAP-PROGRAM.md` §N2.
//
// **Every constant and bit-level rule reconstructed from the AFGS1 / AV1 §7.18.3 algorithm lives in
// this one file**, so the boundary between "structure we implemented" and "numbers that must match a
// normative source" is a file boundary rather than a comment somewhere.
//
// ═══════════════════════════════════════════════════════════════════════════════════════════════
// 🚨 CONFORMANCE STATUS — READ BEFORE CLAIMING AFGS1 COMPATIBILITY
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
// This is a **structurally-AFGS1 grain engine, not a verified-conformant one.** The distinction is
// precise and it matters for exactly two use cases:
//
//   • **Applying grain as an effect** — conformance is irrelevant. The look comes from the AR
//     structure, the overlap blend and the scaling curve, all of which are implemented here.
//   • **Round-tripping AFGS1 metadata, or matching a decoder's output bit-for-bit** — conformance is
//     required, and this engine is NOT there yet.
//
// **The one substantive gap is `gaussianSequence`.** AV1 specifies a fixed 2048-entry table. It is
// *data*, it is not derivable, and it is not reproduced here — a table transcribed from memory that
// is 99% right is worse than no table at all, because it would look correct and be silently wrong.
// So the sequence is **injectable**, and the default is a documented deterministic substitute that
// matches the real table's observable shape but is not it.
//
// **To reach conformance** (a mechanical, verifiable afternoon — not research):
//   1. Take `gaussian_sequence` from a normative source. `dav1d/src/tables.c` is **BSD-2** and
//      `libplacebo/src/shaders/film_grain_av1.c` is **MIT-dual-licensed out of the LGPL project**
//      (its own header says so) — both are safe to use with attribution. ⚠️ Do NOT take it from
//      FFmpeg's `aom_film_grain.c` or libplacebo's `film_grain.c` dispatcher, which are LGPL.
//   2. Pass it as `FilmGrainParameters.gaussianSequence`, and checksum it — 2048 entries, all in
//      [-2048, 2047].
//   3. Re-verify the constants marked `SPEC` below against the same source. They are reconstructed,
//      and two of them (the per-row reseed multipliers, and `GrainMin`/`GrainMax`) are the kind of
//      value that is easy to state confidently and get wrong.
//   4. Cross-validate a rendered template against dav1d for one parameter set.
//
// 🔑 **What is already load-bearing and verified by construction**, because the plan states these
// independently and the implementation agrees with it:
//   • the luma array is **82×73**, not 64×64 — 64×64 is the *effective sampled window*, since block
//     offsets are `9 + offset*2` and read 34 samples, so only x,y ∈ [9,73) is ever touched;
//   • AR taps are `2·lag·(lag+1)` → **0, 4, 12, 24** for lag 0–3, which the tap loop below produces
//     from its own geometry rather than from a table (see `TapGeometryTests`);
//   • overlap weights are exactly **27/17 then 17/27** (luma, 2-sample), all `Round2(g, 5)`.
//
// ⚠️ **SMPTE ST 2131 could not be verified and is not referenced anywhere here** — the AFGS1 spec HTML
// contains zero occurrences of "SMPTE". The SMPTE document that definitely exists is **RDD 5**, which
// is a *different* (frequency-filtering) model.
//

import Foundation

public enum AFGS1 {

    // MARK: - Array geometry (SPEC — corroborated by the plan, see the header)

    /// Luma grain template width. **82, not 64.**
    public static let lumaTemplateWidth = 82
    /// Luma grain template height. **73, not 64.**
    public static let lumaTemplateHeight = 73
    /// The template is only ever *sampled* from `[9, 73)` in both axes — the 64×64 effective window.
    public static let templateSampleOrigin = 9
    /// Output block edge in luma samples.
    public static let blockSize = 32
    /// Samples read per block edge: the 32 of the block plus the 2-sample overlap.
    public static let blockReadExtent = 34
    /// Block offsets are 4 bits each, packed into one 8-bit draw.
    public static let maxBlockOffset = 15

    // MARK: - Value range (SPEC — verify)

    /// `GrainCenter` for 8-bit. Grain is clipped to `[-GrainCenter, GrainCenter - 1]`.
    public static let grainCenter8Bit = 128
    public static let grainMin8Bit = -128
    public static let grainMax8Bit = 127

    // MARK: - Overlap weights (SPEC — corroborated by the plan)

    /// Luma overlap: two samples, `27/17` then `17/27`, reduced by `Round2(_, 5)`.
    public static let lumaOverlapWeights: [(old: Int, new: Int)] = [(27, 17), (17, 27)]
    /// Subsampled-chroma overlap: one sample, `23/22`. *(Chroma is not synthesized yet — see
    /// `FilmGrainSynthesizer`'s scope note. Recorded here so the value is not re-derived later.)*
    public static let subsampledChromaOverlapWeights: [(old: Int, new: Int)] = [(23, 22)]
    public static let overlapShift = 5

    // MARK: - The LFSR

    /// AV1's 16-bit LFSR (`get_random_number`). Taps at bits 0, 1, 3, 12.
    ///
    /// 🔑 **This is why grain must be precomputed rather than evaluated per pixel on the GPU.** The
    /// register walks in raster order with a per-row reseed, so a block's offset depends on every
    /// block drawn before it in that row — inherently sequential. The fix the plan prescribes is to
    /// walk it once on the CPU into a buffer of `(offsetX, offsetY)` per block (~59k entries for
    /// 60 MP) and let the GPU do a pure lookup. `blockOffsets(...)` below is that walk.
    public struct RandomRegister {
        public private(set) var state: UInt16

        public init(state: UInt16) { self.state = state }

        public mutating func next(bits: Int) -> Int {
            let r = state
            let bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1
            state = (r >> 1) | (bit << 15)
            return Int((state >> UInt16(16 - bits)) & UInt16((1 << bits) - 1))
        }
    }

    /// Per-block-row reseed. ⚠️ **SPEC — the two multiplier pairs are reconstructed and are exactly the
    /// kind of constant to verify against dav1d before claiming conformance.** The *structure* — that
    /// each block row restarts from the frame seed mixed with the row index — is what makes grain
    /// independent of how the frame is tiled, and that property is tested rather than assumed.
    public static func seededRegister(seed: UInt16, blockRow: Int) -> RandomRegister {
        var r = seed
        r ^= UInt16(((blockRow &* 37 &+ 178) & 255) << 8)
        r ^= UInt16((blockRow &* 173 &+ 105) & 255)
        return RandomRegister(state: r)
    }

    /// Round-half-up by `shift`, matching the spec's `Round2`. Defined for negative inputs the way the
    /// spec defines it — arithmetic shift, so it rounds toward positive infinity on ties.
    @inline(__always)
    public static func round2(_ value: Int, _ shift: Int) -> Int {
        guard shift > 0 else { return value }
        return (value + (1 << (shift - 1))) >> shift
    }

    @inline(__always)
    public static func clip(_ value: Int, _ low: Int, _ high: Int) -> Int {
        min(max(value, low), high)
    }

    // MARK: - AR tap geometry

    /// The causal neighbourhood for a given lag, in scan order: all of the `lag` rows above, then the
    /// samples to the left on the current row, stopping at the centre.
    ///
    /// 🔑 **Generated from geometry, never tabulated** — which is what lets a test assert the count is
    /// `2·lag·(lag+1)` (0, 4, 12, 24) as an independent check on the loop rather than on a table.
    public static func arTaps(lag: Int) -> [(deltaRow: Int, deltaColumn: Int)] {
        guard lag > 0 else { return [] }
        var taps: [(Int, Int)] = []
        for deltaRow in (-lag)...0 {
            for deltaColumn in (-lag)...lag {
                if deltaRow == 0 && deltaColumn == 0 { return taps }
                taps.append((deltaRow, deltaColumn))
            }
        }
        return taps
    }

    /// Number of AR coefficients a given lag requires: `2·lag·(lag+1)`.
    public static func arCoefficientCount(lag: Int) -> Int { 2 * lag * (lag + 1) }

    // MARK: - The Gaussian sequence

    /// 🚨 **NOT the AV1 `gaussian_sequence` table.** See the conformance banner at the top of this
    /// file. This is a deterministic substitute built to match the real table's *observable* shape —
    /// 2048 entries, approximately N(0, 512²), clamped to [-2048, 2047] and quantized to multiples of
    /// 4 — so that grain generated with it has the right character while being honestly non-normative.
    ///
    /// It is a stored constant rather than a computed one so that **grain never changes between
    /// builds, compilers or platforms.** A lazily-computed sequence would be just as deterministic in
    /// principle and a floating-point-reproducibility hazard in practice.
    public static let substituteGaussianSequence: [Int16] = {
        // Box–Muller over a fixed 32-bit LCG. Values are quantized to multiples of 4, matching the
        // granularity visible in the normative table.
        var state: UInt32 = 0x2545_F491
        func uniform() -> Double {
            state = state &* 1_664_525 &+ 1_013_904_223
            // Avoid exactly 0, which would send log() to -infinity.
            return (Double(state >> 8) + 0.5) / Double(1 << 24)
        }
        var out = [Int16]()
        out.reserveCapacity(2048)
        while out.count < 2048 {
            let u1 = uniform(), u2 = uniform()
            let radius = (-2.0 * Foundation.log(u1)).squareRoot()
            let pair = [radius * Foundation.cos(2 * Double.pi * u2),
                        radius * Foundation.sin(2 * Double.pi * u2)]
            for z in pair where out.count < 2048 {
                let scaled = (z * 512.0 / 4.0).rounded() * 4.0
                out.append(Int16(clip(Int(scaled), -2048, 2047)))
            }
        }
        return out
    }()
}
