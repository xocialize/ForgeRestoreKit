//
// PixelBufferBridge.swift — RestoreTier0
//
// `CVPixelBuffer` ↔ `MTLTexture`, zero-copy, via `CVMetalTextureCache`.
//
// 🔑 **This is the currency the apps actually speak.** `CLAUDE.md`: *"keep pixels GPU-resident. The
// pipeline + preview currency is `CVPixelBuffer` backed by `IOSurface` — **not** `CGImage`, which
// forces a CPU copy."* The `[Float]`-plane APIs elsewhere in this target are the reference/oracle
// layer; an app driving an `MTKView` from `MTLTexture`s must never be routed through them, because a
// chain of `texture → CPU plane → GPU kernel → CPU plane → texture` is **slower than staying on the
// CPU entirely**. Accelerating one stage in isolation is a pessimization; the whole chain has to stay
// resident, which is what `MetalSharpenPipeline` does.
//
// 🚨 **Two format traps, both silent.**
//
//   1. **Never map a buffer as an `_srgb` pixel format.** Metal linearizes on read from an sRGB
//      texture, so the shader would receive linear values while every constant in this package —
//      the noise floor, the coring threshold, the detail scale — is calibrated against *gamma-encoded*
//      luma (N1's own reasoning: sensor noise is signal-dependent, and gamma is approximately
//      variance-stabilizing, which is what lets one global σ work at all). The result is not a crash;
//      it is sharpening that eats shadow detail and leaves highlight noise.
//   2. **Biplanar 4:2:0 is the *better* input, not the harder one.** Its luma is already a separate
//      plane, so "chroma is never touched" stops being a discipline and becomes a fact of the layout —
//      we read plane 0, write plane 0, and never bind plane 1 at all.
//

import CoreVideo
import Foundation
import Metal

public enum PixelBufferBridge {

    public enum BridgeError: Error, Equatable {
        case unsupportedPixelFormat(OSType)
        case textureCacheCreationFailed
        case textureCreationFailed
        case notBackedByIOSurface
    }

    /// The layouts this package can sharpen.
    public enum Layout: Sendable, Equatable {
        /// Packed BGRA. Luma is derived per pixel and the result is applied as an equal delta to R, G
        /// and B — which leaves the colour differences (R−Y, B−Y) untouched, i.e. chroma is preserved
        /// by construction rather than by care.
        case packedBGRA
        /// Biplanar Y′CbCr. Only plane 0 is read or written.
        case biplanarLuma

        public static func detect(_ pixelBuffer: CVPixelBuffer) throws -> Layout {
            switch CVPixelBufferGetPixelFormatType(pixelBuffer) {
            case kCVPixelFormatType_32BGRA:
                return .packedBGRA
            case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                 kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                 kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
                 kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
                return .biplanarLuma
            case let other:
                throw BridgeError.unsupportedPixelFormat(other)
            }
        }
    }

    /// Make a texture cache. Hold onto it — creating one per frame defeats the purpose.
    public static func makeTextureCache(device: MTLDevice) throws -> CVMetalTextureCache {
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess,
              let cache else {
            throw BridgeError.textureCacheCreationFailed
        }
        return cache
    }

    /// Wrap one plane of a pixel buffer as a texture. **No copy** — the texture aliases the buffer's
    /// `IOSurface`, so writes through it are visible to whoever else holds the buffer.
    ///
    /// - Important: the returned `CVMetalTexture` must outlive every use of the `MTLTexture` it
    ///   vends. That is why it is returned rather than discarded: dropping it while the `MTLTexture`
    ///   is still in flight is a use-after-free that usually *appears* to work.
    public static func texture(from pixelBuffer: CVPixelBuffer,
                               plane: Int,
                               pixelFormat: MTLPixelFormat,
                               cache: CVMetalTextureCache) throws -> (CVMetalTexture, MTLTexture) {
        guard CVPixelBufferGetIOSurface(pixelBuffer) != nil else {
            throw BridgeError.notBackedByIOSurface
        }
        let planar = CVPixelBufferIsPlanar(pixelBuffer)
        let width = planar ? CVPixelBufferGetWidthOfPlane(pixelBuffer, plane)
                           : CVPixelBufferGetWidth(pixelBuffer)
        let height = planar ? CVPixelBufferGetHeightOfPlane(pixelBuffer, plane)
                            : CVPixelBufferGetHeight(pixelBuffer)

        var wrapper: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pixelBuffer, nil,
            pixelFormat, width, height, plane, &wrapper)
        guard status == kCVReturnSuccess,
              let wrapper,
              let texture = CVMetalTextureGetTexture(wrapper) else {
            throw BridgeError.textureCreationFailed
        }
        return (wrapper, texture)
    }

    /// The Metal format to map a plane with. **Deliberately never an `_srgb` variant** — see trap 1.
    public static func pixelFormat(for layout: Layout, pixelBuffer: CVPixelBuffer) -> MTLPixelFormat {
        switch layout {
        case .packedBGRA:
            return .bgra8Unorm
        case .biplanarLuma:
            switch CVPixelBufferGetPixelFormatType(pixelBuffer) {
            case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
                 kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
                return .r16Unorm
            default:
                return .r8Unorm
            }
        }
    }

    /// An IOSurface-backed buffer matching `source`'s format and size — the destination for a
    /// non-in-place pass.
    public static func makeCompatibleBuffer(like source: CVPixelBuffer) -> CVPixelBuffer? {
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            kCVPixelBufferMetalCompatibilityKey: true,
        ]
        var destination: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            CVPixelBufferGetWidth(source),
            CVPixelBufferGetHeight(source),
            CVPixelBufferGetPixelFormatType(source),
            attributes as CFDictionary,
            &destination)
        guard status == kCVReturnSuccess else { return nil }

        // Chroma is never written by this package, so a biplanar destination must inherit the source's
        // chroma rather than start black — otherwise "we only touch luma" produces a grey image.
        if let destination, CVPixelBufferIsPlanar(source) {
            copyPlanes(from: source, to: destination, startingAt: 1)
        }
        return destination
    }

    /// Byte-copy planes `index...` — used to carry untouched chroma across to a fresh buffer.
    static func copyPlanes(from source: CVPixelBuffer, to destination: CVPixelBuffer, startingAt index: Int) {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        let count = CVPixelBufferGetPlaneCount(source)
        guard index < count, CVPixelBufferGetPlaneCount(destination) == count else { return }
        for plane in index..<count {
            guard let from = CVPixelBufferGetBaseAddressOfPlane(source, plane),
                  let to = CVPixelBufferGetBaseAddressOfPlane(destination, plane) else { continue }
            let sourceStride = CVPixelBufferGetBytesPerRowOfPlane(source, plane)
            let destinationStride = CVPixelBufferGetBytesPerRowOfPlane(destination, plane)
            let rows = CVPixelBufferGetHeightOfPlane(source, plane)
            let bytes = min(sourceStride, destinationStride)
            for row in 0..<rows {
                memcpy(to.advanced(by: row * destinationStride),
                       from.advanced(by: row * sourceStride),
                       bytes)
            }
        }
    }
}
