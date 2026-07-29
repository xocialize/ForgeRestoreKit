//
// NoiseEstimate.swift — RestoreTier0
//
// Immerkaer's single-pass noise estimator — the σ that every other stage of N1 is expressed in terms
// of. See `mlxengine-todo/GAP-PROGRAM.md` §N1.
//

import Foundation

public enum NoiseEstimate {

    /// Immerkaer (1996): convolve with the 3×3 kernel below, take the mean absolute response, scale by
    /// `√(π/2) / 6`.
    ///
    /// 🔑 **Why this kernel and not a Laplacian.** `[[1,−2,1],[−2,4,−2],[1,−2,1]]` is the difference of
    /// two Laplacians, constructed so it annihilates any locally-**quadratic** surface. Real image
    /// content is locally smooth to second order, so it contributes almost nothing to the response,
    /// while i.i.d. noise passes through with a known gain — which is what makes a *single pass* over
    /// the whole image a defensible estimator of σ instead of a measure of how much detail the picture
    /// happens to contain.
    ///
    /// The `√(π/2)` is the mean-absolute-deviation → standard-deviation conversion for a Gaussian
    /// (E|X| = σ√(2/π)); the `6` is the kernel's L2 norm (√(1+4+1+4+16+4+1+4+1) = 6).
    ///
    /// ⚠️ **Run this on gamma-encoded luma, not linear.** Sensor noise is signal-dependent
    /// (σ² = a·Y + b), so in linear light a single σ is far too large in the shadows and far too small
    /// in the highlights. Gamma encoding is approximately variance-stabilizing, which is precisely what
    /// makes one global number usable. Getting this wrong does not produce an error — it produces
    /// sharpening that eats shadow detail and leaves highlight noise, which reads as "the algorithm is
    /// bad".
    public static func immerkaer(_ plane: [Float], width: Int, height: Int) -> Float {
        precondition(plane.count == width * height)
        guard width >= 3, height >= 3 else { return 0 }

        var total = 0.0
        var count = 0
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let i = y * width + x
                let response =
                    Double(plane[i - width - 1]) - 2 * Double(plane[i - width]) + Double(plane[i - width + 1])
                    - 2 * Double(plane[i - 1]) + 4 * Double(plane[i]) - 2 * Double(plane[i + 1])
                    + Double(plane[i + width - 1]) - 2 * Double(plane[i + width]) + Double(plane[i + width + 1])
                total += abs(response)
                count += 1
            }
        }
        guard count > 0 else { return 0 }
        return Float(total / Double(count) * (Double.pi / 2).squareRoot() / 6.0)
    }
}
