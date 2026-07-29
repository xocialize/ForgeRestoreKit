//
// FilmGrainSynthesizer.swift — RestoreTier0
//
// Applies a `FilmGrainTemplate` to a picture, block by block, with overlap blending and the
// brightness-dependent scaling curve.
//
// ⚠️ **Scope: luma only.** That fully covers the *Gaussian* and *Grey* presets (Grey is literally
// `num_cb_points = num_cr_points = 0`), and covers *Silver Rich* apart from its chroma component.
// Chroma synthesis needs the 44×38 arrays, the extra luma-cross AR tap and subsampling handling —
// more reconstructed spec surface, and it is deliberately not guessed at here. The chroma overlap
// weights are already recorded in `AFGS1Spec` so they are not re-derived when that lands.
//
// 🔑 **The architecture note that makes this affordable on a big image**, and the reason the API is
// shaped around `blockOffsets` rather than a single `apply(image:)` call: the LFSR walk is inherently
// sequential (raster order, per-row reseed), but *everything after it is a pure lookup*. So the walk
// happens once on the CPU into a small buffer — ~59k `(offsetX, offsetY)` pairs for 60 MP — and the
// per-pixel stage becomes one dispatch over output pixels with no atomics and no barriers. The plan's
// estimate for that shape is sub-millisecond for 24 MP on an M1 Pro. The CPU path below is the
// reference implementation and the correctness oracle for that dispatch.
//

import Foundation

public struct FilmGrainSynthesizer: Sendable {

    public let parameters: FilmGrainParameters
    public let template: FilmGrainTemplate
    private let scalingLUT: [Int]

    public init(parameters: FilmGrainParameters) throws {
        try parameters.validate()
        self.parameters = parameters
        self.template = .luma(parameters: parameters)
        self.scalingLUT = parameters.scalingLUT()
    }

    /// A block's sampling origin into the template.
    public struct BlockOffset: Sendable, Equatable {
        public var x: Int
        public var y: Int
        public init(x: Int, y: Int) { self.x = x; self.y = y }
    }

    /// Offsets for one **absolute** block row, columns `0...lastColumn`.
    ///
    /// 🔑 **Absolute, never tile-relative — this is the property the whole feature rests on.** Block
    /// (3, 7) must draw the same offsets whether it was reached from a full-frame render or from a
    /// viewport tile starting halfway down the image, or grain crawls under the picture as the user
    /// pans. Note the cost this imposes and why it is unavoidable: the LFSR is sequential within a
    /// row, so a tile starting at column 40 must still walk columns 0…39 to arrive at the right state.
    /// The reseed is per *row*, so that walk is bounded by one row, not by the whole frame.
    public func blockOffsets(blockRow: Int, throughColumn lastColumn: Int) -> [BlockOffset] {
        guard lastColumn >= 0 else { return [] }
        var register = AFGS1.seededRegister(seed: parameters.seed, blockRow: blockRow)
        return (0...lastColumn).map { _ in
            let draw = register.next(bits: 8)
            return BlockOffset(x: draw >> 4, y: draw & 15)
        }
    }

    /// Convenience: offsets for every block covering a `width × height` picture at the origin.
    public func blockOffsets(width: Int, height: Int) -> [[BlockOffset]] {
        let columns = (width + AFGS1.blockSize - 1) / AFGS1.blockSize
        let rows = (height + AFGS1.blockSize - 1) / AFGS1.blockSize
        guard columns > 0, rows > 0 else { return [] }
        return (0..<rows).map { blockOffsets(blockRow: $0, throughColumn: columns - 1) }
    }

    /// The raw grain field for a picture, before it meets any pixels — block sampling plus the overlap
    /// blend. Exposed separately from `apply` because it is the half a GPU dispatch would replace, and
    /// because the seam-continuity property is far easier to assert on grain than through a picture.
    ///
    /// 🔑 **Why the overlap partner is read from the neighbour's template continuation, never from what
    /// the neighbour wrote.** This is the detail the plan's *"block offsets are `9 + offsetX*2` reading
    /// **34** samples"* is really about, and it is easy to get wrong in a way that looks plausible:
    /// adjacent 32-px blocks do **not** overlap in output space — block `c` covers columns
    /// `[32c, 32c+32)` and block `c−1` covers `[32(c−1), 32c)`. So blending against the output buffer
    /// blends against *zero*, which does not join two grain fields, it just attenuates a 2-px stripe —
    /// a seam that is quieter instead of absent. The spec reads 34 samples precisely so each block has
    /// a 2-sample **continuation** past its own 32, and that continuation is the partner. Reading it
    /// from the template also makes the field independent of write order, which is what keeps tiled and
    /// full-frame renders identical.
    ///
    /// ⚠️ **SPEC — verify:** the corner ordering. Vertical is applied before horizontal here, so the
    /// 2×2 corner where both seams meet is blended in that sequence. The spec sequences it too; which
    /// order is normative should be confirmed against dav1d alongside the other `SPEC` items.
    /// - Parameters:
    ///   - width/height: the extent to produce.
    ///   - originX/originY: where that extent sits in the **full picture**, in pixels. A tiled renderer
    ///     passes the tile's origin; a full-frame render passes `(0, 0)`. Need not be block-aligned.
    public func grainField(width: Int, height: Int,
                           originX: Int = 0, originY: Int = 0) -> [Int] {
        var field = [Int](repeating: 0, count: max(0, width * height))
        guard width > 0, height > 0 else { return field }
        let size = AFGS1.blockSize
        let origin = AFGS1.templateSampleOrigin

        let firstBlockRow = originY / size
        let lastBlockRow = (originY + height - 1) / size
        let firstBlockColumn = originX / size
        let lastBlockColumn = (originX + width - 1) / size

        // One extra row above, so a tile whose top edge sits inside the picture still has the
        // neighbour its top seam blends against. Columns always walk from 0 (the LFSR requires it).
        var offsetsByRow: [Int: [BlockOffset]] = [:]
        for row in max(0, firstBlockRow - 1)...lastBlockRow {
            offsetsByRow[row] = blockOffsets(blockRow: row, throughColumn: lastBlockColumn)
        }

        @inline(__always)
        func sample(_ offset: BlockOffset, _ x: Int, _ y: Int) -> Int {
            template[origin + offset.x * 2 + x, origin + offset.y * 2 + y]
        }

        for blockRow in firstBlockRow...lastBlockRow {
            guard let row = offsetsByRow[blockRow] else { continue }
            for blockColumn in firstBlockColumn...lastBlockColumn {
                let offset = row[blockColumn]
                let baseX = blockColumn * size
                let baseY = blockRow * size

                for y in 0..<size {
                    let pixelY = baseY + y - originY
                    guard pixelY >= 0 else { continue }
                    guard pixelY < height else { break }
                    for x in 0..<size {
                        let pixelX = baseX + x - originX
                        guard pixelX >= 0 else { continue }
                        guard pixelX < width else { break }

                        var value = sample(offset, x, y)

                        if parameters.overlap {
                            // Top seam — partner is the block above, at its continuation rows 32+y.
                            if blockRow > 0, y < AFGS1.lumaOverlapWeights.count,
                               let aboveRow = offsetsByRow[blockRow - 1] {
                                let w = AFGS1.lumaOverlapWeights[y]
                                let above = aboveRow[blockColumn]
                                let old = sample(above, x, size + y)
                                value = AFGS1.clip(AFGS1.round2(old * w.old + value * w.new,
                                                                AFGS1.overlapShift),
                                                   AFGS1.grainMin8Bit, AFGS1.grainMax8Bit)
                            }
                            // Left seam — partner is the block to the left, continuation columns 32+x.
                            if blockColumn > 0, x < AFGS1.lumaOverlapWeights.count {
                                let w = AFGS1.lumaOverlapWeights[x]
                                let left = row[blockColumn - 1]
                                var old = sample(left, size + x, y)
                                // The partner's own top seam applies to it as well, or the corner
                                // would blend against an unblended neighbour.
                                if blockRow > 0, y < AFGS1.lumaOverlapWeights.count,
                                   let aboveRow = offsetsByRow[blockRow - 1] {
                                    let wy = AFGS1.lumaOverlapWeights[y]
                                    let aboveLeft = aboveRow[blockColumn - 1]
                                    let older = sample(aboveLeft, size + x, size + y)
                                    old = AFGS1.clip(AFGS1.round2(older * wy.old + old * wy.new,
                                                                  AFGS1.overlapShift),
                                                     AFGS1.grainMin8Bit, AFGS1.grainMax8Bit)
                                }
                                value = AFGS1.clip(AFGS1.round2(old * w.old + value * w.new,
                                                                AFGS1.overlapShift),
                                                   AFGS1.grainMin8Bit, AFGS1.grainMax8Bit)
                            }
                        }
                        field[pixelY * width + pixelX] = value
                    }
                }
            }
        }
        return field
    }

    /// Apply grain to a gamma-encoded luma plane in `[0, 1]`.
    ///
    /// Grain amplitude is looked up by the **source** pixel's brightness, not the result's — the curve
    /// describes how much grain a tone should carry, and feeding it the already-grained value would
    /// make the effect depend on its own output.
    public func apply(to plane: [Float], width: Int, height: Int,
                      originX: Int = 0, originY: Int = 0) -> [Float] {
        precondition(plane.count == width * height)
        guard !scalingLUT.allSatisfy({ $0 == 0 }) else { return plane }

        let field = grainField(width: width, height: height, originX: originX, originY: originY)
        var out = plane
        for i in 0..<plane.count {
            let level = AFGS1.clip(Int((plane[i] * 255).rounded()), 0, 255)
            let scaled = AFGS1.round2(scalingLUT[level] * field[i], parameters.grainScalingShift)
            out[i] = Float(min(max((plane[i] * 255) + Float(scaled), 0), 255)) / 255
        }
        return out
    }
}
