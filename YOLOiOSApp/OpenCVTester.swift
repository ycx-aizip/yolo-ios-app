import Foundation
import UIKit

class OpenCVTester {
    static func testOpenCVIntegration() -> Bool {
        let result = OpenCVBridge.isOpenCVWorking()
        
        if result {
            print("✅ OpenCV is working correctly!")
        } else {
            print("❌ There's an issue with OpenCV integration")
        }
        
        return result
    }
} 