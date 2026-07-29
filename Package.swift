// swift-tools-version: 6.2
import PackageDescription

// ForgeRestoreKit — pixel quality at constant resolution (FORGE-PRODUCT-ARCHITECTURE.md §4.5).
//
// ⚠️ **Deliberately partial.** This repo exists today for **`RestoreTier0` only** — the net-clean
// classical half that §4.5 reserves for N1 (noise-aware sharpen), **N2 (AFGS1 film grain)** and N3
// (hot/dead pixel). The umbrella product, the `FrameRestoreProvider` seam and the `RestoreTier2`
// engine-backed half are `R2-HANDOFF.md` **BRIDGE-070**, and they are held back on purpose: 070 is
// gated on decisions **D1** (one repo per family vs monorepo) and **D2** (the restore job vocabulary —
// denoise / motionDeblur / defocusDeblur become public API the moment they are tagged). Nothing here
// names a job, so neither decision is preempted by this package existing.
//
// Category A: no MLX, no engine, no vendored binaries, macOS-14 floor, no Metal requirement.
let package = Package(
    name: "ForgeRestoreKit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "RestoreTier0", targets: ["RestoreTier0"]),
    ],
    dependencies: [
        // The shared CVPixelBuffer <-> MTLTexture currency. Extracted from this target after a
        // duplicated copy in a sibling Kit diverged and collided for any consumer importing both.
        .package(url: "https://github.com/xocialize/ForgePixelBridge.git", from: "0.1.0"),
    ],
    targets: [
        .target(name: "RestoreTier0",
                dependencies: [.product(name: "ForgePixelBridge", package: "ForgePixelBridge")]),
        .testTarget(name: "RestoreTier0Tests",
                    dependencies: ["RestoreTier0",
                                   .product(name: "ForgePixelBridge", package: "ForgePixelBridge")]),
    ]
)
