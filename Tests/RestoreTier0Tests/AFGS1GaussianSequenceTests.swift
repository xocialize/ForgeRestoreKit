// The normative table's integrity gate. A corrupted or re-typed table would look correct and be
// silently wrong (the AFGS1Spec banner's exact warning), so the observable shape AND spot values
// are pinned against the verified dav1d transcription.
import XCTest

@testable import RestoreTier0

final class AFGS1GaussianSequenceTests: XCTestCase {

    func testNormativeTableShape() {
        let t = AFGS1.gaussianSequence
        XCTAssertEqual(t.count, 2048)
        XCTAssertTrue(t.allSatisfy { $0 % 4 == 0 }, "every normative entry is a multiple of 4")
        XCTAssertTrue(t.allSatisfy { (-2048...2047).contains(Int($0)) })
        let mean = Double(t.reduce(0) { $0 + Int($1) }) / 2048
        let sd = (t.reduce(0.0) { $0 + pow(Double($1) - mean, 2) } / 2048).squareRoot()
        XCTAssertEqual(mean, 0.546875, accuracy: 0.0001)   // exact: 1120/2048
        XCTAssertEqual(sd, 511.5, accuracy: 0.1)
    }

    func testSpotValuesMatchTheVerifiedTranscription() {
        let t = AFGS1.gaussianSequence
        XCTAssertEqual(Array(t.prefix(8)), [56, 568, -180, 172, 124, -84, 172, -64])
        XCTAssertEqual(Array(t.suffix(4)), [288, 944, 428, -484])
    }

    func testNormativeIsTheDefaultAndSubstituteSurvives() {
        XCTAssertEqual(FilmGrainParameters().gaussianSequence, AFGS1.gaussianSequence)
        // The substitute remains for reproducing pre-0.11.0 grain fields; they must differ.
        XCTAssertNotEqual(AFGS1.substituteGaussianSequence, AFGS1.gaussianSequence)
        XCTAssertEqual(AFGS1.substituteGaussianSequence.count, 2048)
    }
}
