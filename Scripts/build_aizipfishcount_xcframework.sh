#!/bin/bash
set -e

# Build AizipFishCount.xcframework for partner distribution
# This script compiles AizipFishCount as a universal binary framework
# including both device (arm64) and simulator (arm64 + x86_64) architectures

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_PROJECT="${PROJECT_ROOT}/AizipFishCountApp-Release"
BUILD_DIR="${RELEASE_PROJECT}/Library/"
SCHEME="AizipFishCount"

echo "🔨 Building AizipFishCount.xcframework..."
echo "Project root: ${PROJECT_ROOT}"
echo "Release project: ${RELEASE_PROJECT}"
echo ""

# Check if Release project exists
if [ ! -d "${RELEASE_PROJECT}" ]; then
    echo "❌ Error: Release project not found at: ${RELEASE_PROJECT}"
    echo "Please ensure AizipFishCountApp-Release exists"
    exit 1
fi

# Sync source code from Sources/AizipFishCount to Release project
echo "🔄 Syncing source code from Sources/AizipFishCount..."
SOURCE_DIR="${PROJECT_ROOT}/Sources/AizipFishCount"
FRAMEWORK_SOURCE="${RELEASE_PROJECT}/AizipFishCount"

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "❌ Error: Source directory not found at: ${SOURCE_DIR}"
    exit 1
fi

# Remove old framework source and copy fresh
rm -rf "${FRAMEWORK_SOURCE}"
cp -R "${SOURCE_DIR}" "${FRAMEWORK_SOURCE}"
echo "✅ Source code synced successfully"
echo ""

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Build for iOS Device (arm64)
echo "📱 Building for iOS Device (arm64)..."
xcodebuild archive \
    -project "${RELEASE_PROJECT}/AizipFishCountApp.xcodeproj" \
    -scheme "${SCHEME}" \
    -destination "generic/platform=iOS" \
    -archivePath "${BUILD_DIR}/AizipFishCount-iOS.xcarchive" \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGN_IDENTITY="" \
    CODE_SIGNING_ALLOWED=NO

# Build for iOS Simulator (arm64 + x86_64)
echo "💻 Building for iOS Simulator (arm64 + x86_64)..."
xcodebuild archive \
    -project "${RELEASE_PROJECT}/AizipFishCountApp.xcodeproj" \
    -scheme "${SCHEME}" \
    -destination "generic/platform=iOS Simulator" \
    -archivePath "${BUILD_DIR}/AizipFishCount-iOS-Simulator.xcarchive" \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGN_IDENTITY="" \
    CODE_SIGNING_ALLOWED=NO

# Create XCFramework
echo "📦 Creating XCFramework..."
xcodebuild -create-xcframework \
    -framework "${BUILD_DIR}/AizipFishCount-iOS.xcarchive/Products/Library/Frameworks/AizipFishCount.framework" \
    -framework "${BUILD_DIR}/AizipFishCount-iOS-Simulator.xcarchive/Products/Library/Frameworks/AizipFishCount.framework" \
    -output "${BUILD_DIR}/AizipFishCount.xcframework"

# Show size
echo ""
echo "✅ XCFramework built successfully!"
echo "📍 Location: ${BUILD_DIR}/AizipFishCount.xcframework"
echo "📊 Size: $(du -sh "${BUILD_DIR}/AizipFishCount.xcframework" | cut -f1)"
echo ""
echo "Contents:"
ls -lh "${BUILD_DIR}/AizipFishCount.xcframework"
