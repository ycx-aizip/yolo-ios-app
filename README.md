# Fish Counting iOS SDK — Development Repository

Swift development repository for the Fish Counting iOS SDK. Contains the SDK source (backend + UI), build scripts, and an example app. The backend compiles to a binary xcframework; the UI ships as source. Intended for Aizip/SoftBank developers.

## Structure

```
yolo-ios-app/
├── Sources/
│   ├── AizipFishCount/   # Backend — tracking, detection, counting (→ xcframework)
│   └── Visualization/    # UI components (→ distributed as source)
├── AizipFishCountApp/    # Example app (Swift Package Manager)
├── Packages/opencv2/     # OpenCV dependency (app-level)
└── Scripts/              # Build & release automation
```

## Usage

- **Requirements**: Xcode 16+, iOS 16+, iPad Pro/Air recommended.
- **Develop & test**: open `AizipFishCountApp/AizipFishCountApp.xcodeproj` (SPM imports the source directly for fast incremental builds).
- **Build & release**:
  ```bash
  ./Scripts/build_aizipfishcount_xcframework.sh   # build binary framework
  ./Scripts/release.sh                            # package to ../Aizip_softbank_fishcount_ipad/
  ```
- **Contributing**: read the Python reference in `../vision_yolo/` first and maintain algorithm parity; open a pull request. Detailed module and algorithm notes live in `../docs/`.

## License

Proprietary — Aizip Inc. & SoftBank
