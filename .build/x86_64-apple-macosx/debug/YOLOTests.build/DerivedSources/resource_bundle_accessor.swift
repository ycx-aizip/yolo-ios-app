import Foundation

extension Foundation.Bundle {
    static let module: Bundle = {
        let mainPath = Bundle.main.bundleURL.appendingPathComponent("YOLO_YOLOTests.bundle").path
        let buildPath = "/Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/.build/x86_64-apple-macosx/debug/YOLO_YOLOTests.bundle"

        let preferredBundle = Bundle(path: mainPath)

        guard let bundle = preferredBundle ?? Bundle(path: buildPath) else {
            // Users can write a function called fatalError themselves, we should be resilient against that.
            Swift.fatalError("could not load resource bundle: from \(mainPath) or \(buildPath)")
        }

        return bundle
    }()
}