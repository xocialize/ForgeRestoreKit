//
// FilmGrainEstimator.swift — RestoreTier0
//
// **"Match the original grain."** `GAP-PROGRAM.md` §N2 calls this the differentiated feature, and the
// reason it is cheap for us is structural: the estimator fits noise to the **residual**, and a restore
// pipeline already produces the denoised image, so `original − denoised` costs nothing. The competitor
// ships three fixed grain presets and has no equivalent.
//
// Three stages, following `aom_dsp/noise_model.c` (whose clean-room Rust port, `av1-grain`, is BSD-2 +
// AOM Patent License 1.0 — this is an independent implementation from the described algorithm):
//
//   1. **Flat-block finding.** Fit the AR model only where the picture has no structure, or the
//      "noise" model learns the *content* instead. Detected from the gradient covariance matrix's
//      eigenvalues on the denoised image.
//   2. **AR coefficients by plain least squares** over the causal neighbourhood of the residual,
//      restricted to those blocks.
//   3. **A scaling curve** — noise amplitude as a function of luma — reduced to ≤14 knees.
//
// 🔑 **Stage 3 is calibrated by measurement, not derived.** The synthesizer's output amplitude is a
// product of the Gaussian source's spread, `grainScaleShift`, the AR filter's gain, the scaling curve
// and `grainScalingShift` — deriving a closed form through all of that is possible and brittle. So the
// estimator **builds the template with the fitted coefficients, measures its RMS, and solves the curve
// against that measurement**. It is the same discipline the rest of this package is built on, and it
// has the useful property of staying correct if the Gaussian source is ever swapped for the normative
// AV1 table (which would change the spread and silently invalidate any derived constant).
//
// ⚠️ **This does not recover the true coefficients, and that is not the goal.** Fitting an AR model to
// one noisy realization is ill-posed — many parameter sets explain the same field. The goal is a
// *perceptual* round trip: synthesize from the estimate and get noise of the same strength and the
// same coarseness. That is what the tests assert, and asserting coefficient recovery instead would be
// measuring the fixture rather than the feature.
//

import Foundation

public enum FilmGrainEstimator {

    public enum EstimationError: Error, Equatable {
        case sizeMismatch
        case tooSmall
        /// Not enough flat area to fit anything. **An honest failure beats a confident wrong answer**:
        /// fitting on structured blocks produces a model of the *picture*, which then gets synthesized
        /// over the picture as "grain".
        case notEnoughFlatBlocks(found: Int, needed: Int)
    }

    public struct Options: Sendable {
        /// Block edge used for flatness scoring.
        public var blockSize: Int
        /// Fraction of the flattest blocks to fit on.
        ///
        /// 🔑 **A percentile, not an absolute threshold.** Flatness magnitude depends on exposure,
        /// content and noise level, so any fixed cutoff is wrong for some images — it either finds no
        /// blocks in a busy frame or accepts textured ones in a smooth frame. Taking the flattest
        /// quartile is scale-free and always yields a workable sample.
        public var flatBlockFraction: Double
        /// Minimum flat blocks before a fit is attempted.
        public var minimumFlatBlocks: Int
        /// AR lag to fit, 1…3.
        public var arCoefficientLag: Int
        /// Ridge term on the normal equations. Small, and load-bearing: the causal neighbourhood of a
        /// correlated field is close to collinear, so an unregularized solve can produce enormous
        /// coefficients that make the synthesizer's recursion run away.
        public var ridge: Double
        /// Luma buckets for the scaling curve, before reduction to ≤14 knees.
        public var lumaBuckets: Int

        public init(blockSize: Int = 32,
                    flatBlockFraction: Double = 0.25,
                    minimumFlatBlocks: Int = 4,
                    arCoefficientLag: Int = 2,
                    ridge: Double = 1e-3,
                    lumaBuckets: Int = 16) {
            self.blockSize = blockSize
            self.flatBlockFraction = flatBlockFraction
            self.minimumFlatBlocks = minimumFlatBlocks
            self.arCoefficientLag = arCoefficientLag
            self.ridge = ridge
            self.lumaBuckets = lumaBuckets
        }

        public static let `default` = Options()
    }

    public struct Estimate: Sendable {
        /// Feed straight to `FilmGrainSynthesizer`.
        public let parameters: FilmGrainParameters
        /// Residual standard deviation over the flat blocks, in [0, 1] units — the overall noise level.
        public let residualSigma: Float
        /// How much of the frame was flat enough to fit on. **Low means treat the fit as weak**, and it
        /// is the honest input to a "match source" confidence indicator.
        public let flatBlockFraction: Double
        public let flatBlockCount: Int
        /// Measured RMS of the template the fitted coefficients produced — the calibration the scaling
        /// curve was solved against.
        public let templateRMS: Float
        /// Nearest-neighbour correlation of the residual: how *coarse* the source grain is.
        public let residualCorrelation: Double
    }

    // MARK: - Fit

    /// Fit a grain model to `original − denoised`.
    ///
    /// - Parameters:
    ///   - original: the noisy source, gamma-encoded luma in [0, 1].
    ///   - denoised: the same frame after denoising. The pipeline already has this.
    public static func fit(original: [Float], denoised: [Float],
                           width: Int, height: Int,
                           seed: UInt16 = 0x1234,
                           options: Options = .default) throws -> Estimate {
        guard original.count == width * height, denoised.count == width * height else {
            throw EstimationError.sizeMismatch
        }
        let block = max(8, options.blockSize)
        guard width >= block, height >= block else { throw EstimationError.tooSmall }

        let residual = (0..<original.count).map { original[$0] - denoised[$0] }

        // ── 1 · Flat blocks ───────────────────────────────────────────────────────────────────
        let blocks = flatBlocks(in: denoised, width: width, height: height, options: options)
        guard blocks.count >= options.minimumFlatBlocks else {
            throw EstimationError.notEnoughFlatBlocks(found: blocks.count,
                                                      needed: options.minimumFlatBlocks)
        }
        let columns = width / block, rows = height / block
        let fraction = Double(blocks.count) / Double(max(1, columns * rows))

        // ── 2 · AR coefficients by least squares over the causal neighbourhood ────────────────
        let lag = min(3, max(1, options.arCoefficientLag))
        let taps = AFGS1.arTaps(lag: lag)
        let solved = leastSquaresAR(residual: residual, width: width, blocks: blocks,
                                    blockSize: block, taps: taps, ridge: options.ridge)

        // Quantize to the spec's fixed point. The shift is chosen so the coefficients use as much of
        // the Int8 range as they can without clipping — the alternative, a fixed shift, wastes
        // precision on a weak fit and clips a strong one.
        let (coefficients, shift) = quantize(solved)

        // ── 3 · The scaling curve, calibrated against the template these coefficients produce ──
        var probe = FilmGrainParameters(seed: seed,
                                        arCoefficientLag: lag,
                                        arCoefficients: coefficients,
                                        arCoefficientShift: shift,
                                        grainScaleShift: 0,
                                        grainScalingShift: 8,
                                        scalingPoints: [ScalingPoint(value: 0, scaling: 255)],
                                        overlap: true,
                                        amount: 1.0)
        let template = FilmGrainTemplate.luma(parameters: probe)
        let templateRMS = rms(template.samples.map { Double($0) })

        let (points, sigma) = scalingCurve(residual: residual, guide: denoised, width: width,
                                           blocks: blocks, blockSize: block,
                                           templateRMS: templateRMS,
                                           grainScalingShift: probe.grainScalingShift,
                                           buckets: options.lumaBuckets)
        probe.scalingPoints = points

        return Estimate(parameters: probe,
                        residualSigma: Float(sigma),
                        flatBlockFraction: fraction,
                        flatBlockCount: blocks.count,
                        templateRMS: Float(templateRMS),
                        residualCorrelation: correlation(residual: residual, width: width,
                                                         blocks: blocks, blockSize: block))
    }

    // MARK: - 1 · Flat-block finding

    /// Blocks with the least structure, as `(x, y)` pixel origins.
    ///
    /// Structure is measured from the **gradient covariance matrix** `[[Σgx², Σgxgy], [Σgxgy, Σgy²]]`.
    /// Its larger eigenvalue is the energy along the dominant edge direction, so it is large for an
    /// edge *and* for texture, while staying small on genuinely flat area — which is exactly the
    /// discrimination needed. Scoring on the trace alone would rank a strong single edge the same as
    /// isotropic texture of equal energy.
    static func flatBlocks(in guide: [Float], width: Int, height: Int,
                           options: Options) -> [(x: Int, y: Int)] {
        let block = max(8, options.blockSize)
        let columns = width / block, rows = height / block
        guard columns > 0, rows > 0 else { return [] }

        var scored: [(score: Double, x: Int, y: Int)] = []
        scored.reserveCapacity(columns * rows)

        for row in 0..<rows {
            for column in 0..<columns {
                let ox = column * block, oy = row * block
                var gxx = 0.0, gyy = 0.0, gxy = 0.0
                for y in (oy + 1)..<(oy + block - 1) {
                    for x in (ox + 1)..<(ox + block - 1) {
                        let i = y * width + x
                        let gx = Double(guide[i + 1] - guide[i - 1]) * 0.5
                        let gy = Double(guide[i + width] - guide[i - width]) * 0.5
                        gxx += gx * gx; gyy += gy * gy; gxy += gx * gy
                    }
                }
                let trace = gxx + gyy
                let determinant = gxx * gyy - gxy * gxy
                let discriminant = max(0, trace * trace - 4 * determinant)
                let larger = (trace + discriminant.squareRoot()) / 2
                scored.append((larger, ox, oy))
            }
        }

        scored.sort { $0.score < $1.score }
        let wanted = max(options.minimumFlatBlocks,
                         Int((Double(scored.count) * options.flatBlockFraction).rounded()))
        return scored.prefix(min(wanted, scored.count)).map { (x: $0.x, y: $0.y) }
    }

    // MARK: - 2 · Least squares

    /// Solve `Σ a_k · r[n−k] ≈ r[n]` over the flat blocks — the AR model of the residual.
    static func leastSquaresAR(residual: [Float], width: Int,
                               blocks: [(x: Int, y: Int)], blockSize: Int,
                               taps: [(deltaRow: Int, deltaColumn: Int)],
                               ridge: Double) -> [Double] {
        let count = taps.count
        guard count > 0 else { return [] }
        var normal = [Double](repeating: 0, count: count * count)   // XᵀX
        var target = [Double](repeating: 0, count: count)           // Xᵀy
        let margin = taps.map { max(abs($0.deltaRow), abs($0.deltaColumn)) }.max() ?? 1

        var neighbours = [Double](repeating: 0, count: count)
        for block in blocks {
            for y in (block.y + margin)..<(block.y + blockSize - margin) {
                for x in (block.x + margin)..<(block.x + blockSize - margin) {
                    let centre = Double(residual[y * width + x])
                    for (i, tap) in taps.enumerated() {
                        neighbours[i] = Double(residual[(y + tap.deltaRow) * width + x + tap.deltaColumn])
                    }
                    for i in 0..<count {
                        target[i] += neighbours[i] * centre
                        for j in i..<count {
                            normal[i * count + j] += neighbours[i] * neighbours[j]
                        }
                    }
                }
            }
        }
        // Mirror the upper triangle and add the ridge.
        for i in 0..<count {
            normal[i * count + i] += ridge * max(1e-12, normal[i * count + i])
            for j in (i + 1)..<count { normal[j * count + i] = normal[i * count + j] }
        }
        return solveSymmetric(normal, target, count) ?? [Double](repeating: 0, count: count)
    }

    /// Cholesky solve for a symmetric positive-definite system. Returns `nil` if the matrix is not
    /// positive definite, which the caller treats as "no usable fit" rather than pressing on.
    static func solveSymmetric(_ matrix: [Double], _ rhs: [Double], _ n: Int) -> [Double]? {
        var l = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0...i {
                var sum = matrix[i * n + j]
                for k in 0..<j { sum -= l[i * n + k] * l[j * n + k] }
                if i == j {
                    guard sum > 0 else { return nil }
                    l[i * n + i] = sum.squareRoot()
                } else {
                    l[i * n + j] = sum / l[j * n + j]
                }
            }
        }
        var y = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var sum = rhs[i]
            for k in 0..<i { sum -= l[i * n + k] * y[k] }
            y[i] = sum / l[i * n + i]
        }
        var x = [Double](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var sum = y[i]
            for k in (i + 1)..<n { sum -= l[k * n + i] * x[k] }
            x[i] = sum / l[i * n + i]
        }
        return x
    }

    /// Real coefficients → the spec's `Int8` fixed point, choosing the shift that uses the most of the
    /// available range without clipping.
    ///
    /// ⚠️ **Total feedback is capped below 1.0 in filter terms.** An AR fit on a noisy realization can
    /// come back with a gain at or above unity, and the synthesizer's recursion would then run away
    /// and clip the whole template to ±127 — grain that is a flat field with a histogram that still
    /// looks plausible. Scaling the whole vector preserves its *shape* (the coarseness) while making
    /// it stable.
    static func quantize(_ coefficients: [Double]) -> ([Int8], Int) {
        guard !coefficients.isEmpty else { return ([], 6) }
        var values = coefficients
        let total = values.reduce(0) { $0 + abs($1) }
        let ceiling = 0.92
        if total > ceiling { values = values.map { $0 * ceiling / total } }

        for shift in 6...9 {
            let scale = Double(1 << shift)
            let scaled = values.map { ($0 * scale).rounded() }
            if scaled.allSatisfy({ $0 >= -128 && $0 <= 127 }) {
                return (scaled.map { Int8($0) }, shift)
            }
        }
        let scale = Double(1 << 9)
        return (values.map { Int8(AFGS1.clip(Int(($0 * scale).rounded()), -128, 127)) }, 9)
    }

    // MARK: - 3 · The scaling curve

    /// Residual σ per luma bucket → ≤14 scaling knees, calibrated against the measured template RMS.
    static func scalingCurve(residual: [Float], guide: [Float], width: Int,
                             blocks: [(x: Int, y: Int)], blockSize: Int,
                             templateRMS: Double, grainScalingShift: Int,
                             buckets: Int) -> ([ScalingPoint], Double) {
        let buckets = max(2, min(14, buckets))
        var sums = [Double](repeating: 0, count: buckets)
        var squares = [Double](repeating: 0, count: buckets)
        var counts = [Double](repeating: 0, count: buckets)

        for block in blocks {
            for y in block.y..<(block.y + blockSize) {
                for x in block.x..<(block.x + blockSize) {
                    let i = y * width + x
                    let level = AFGS1.clip(Int(guide[i] * Float(buckets)), 0, buckets - 1)
                    let r = Double(residual[i])
                    sums[level] += r; squares[level] += r * r; counts[level] += 1
                }
            }
        }

        // Overall σ, for the report and as the fallback for empty buckets.
        let totalCount = counts.reduce(0, +)
        let totalSum = sums.reduce(0, +), totalSquares = squares.reduce(0, +)
        let overall = totalCount > 1
            ? max(0, totalSquares / totalCount - pow(totalSum / totalCount, 2)).squareRoot()
            : 0

        // ⚠️ The synthesizer computes `Round2(lut[level] · grain, shift)`, so with the template's RMS
        // measured, the curve that hits a target σ is a division — no derivation through the Gaussian
        // source, the AR gain or the scale shift, all of which would have to be re-derived if any of
        // them changed.
        let scale = Double(1 << grainScalingShift)
        var points: [ScalingPoint] = []
        for bucket in 0..<buckets {
            let n = counts[bucket]
            let sigma: Double
            if n > 16 {
                sigma = max(0, squares[bucket] / n - pow(sums[bucket] / n, 2)).squareRoot()
            } else {
                sigma = overall   // too few samples to trust — fall back rather than invent a knee
            }
            let target = sigma * 255.0                       // residual is [0,1]; grain is in codes
            let value = templateRMS > 1e-9 ? target * scale / templateRMS : 0
            let luma = Int((Double(bucket) + 0.5) / Double(buckets) * 255)
            points.append(ScalingPoint(value: AFGS1.clip(luma, 0, 255),
                                       scaling: AFGS1.clip(Int(value.rounded()), 0, 255)))
        }
        return (points, overall)
    }

    // MARK: - Small statistics

    static func rms(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.reduce(0.0) { $0 + pow($1 - mean, 2) } / Double(values.count)
        return variance.squareRoot()
    }

    /// Nearest-neighbour horizontal correlation of the residual over the flat blocks — the coarseness
    /// the AR fit is trying to reproduce, and the quantity the round-trip test checks.
    static func correlation(residual: [Float], width: Int,
                            blocks: [(x: Int, y: Int)], blockSize: Int) -> Double {
        var product = 0.0, energy = 0.0
        for block in blocks {
            for y in block.y..<(block.y + blockSize) {
                for x in block.x..<(block.x + blockSize - 1) {
                    let i = y * width + x
                    product += Double(residual[i]) * Double(residual[i + 1])
                    energy += Double(residual[i]) * Double(residual[i])
                }
            }
        }
        return energy > 0 ? product / energy : 0
    }
}
