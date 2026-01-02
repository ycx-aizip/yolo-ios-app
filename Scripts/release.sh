#!/bin/bash
set -e

# Package Fish Counting SDK for partner distribution
# This script copies xcframework, source files, and dependencies to the release repository

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/.build"
RELEASE_REPO="${PROJECT_ROOT}/../Aizip_softbank_fishcount_ipad"

echo "📦 Packaging Fish Counting SDK for distribution..."
echo "Development repo: ${PROJECT_ROOT}"
echo "Release repo: ${RELEASE_REPO}"
echo ""

# Check if release repository exists
if [ ! -d "${RELEASE_REPO}" ]; then
    echo "❌ Error: Release repository not found at: ${RELEASE_REPO}"
    echo "Please ensure Aizip_softbank_fishcount_ipad exists"
    exit 1
fi

# Check if xcframework exists
XCFRAMEWORK="${BUILD_DIR}/AizipFishCount.xcframework"
if [ ! -d "${XCFRAMEWORK}" ]; then
    echo "❌ Error: XCFramework not found at: ${XCFRAMEWORK}"
    echo "Please build it first: ./Scripts/build_aizipfishcount_xcframework.sh"
    exit 1
fi

# Create necessary directories in release repo
echo "📁 Creating directories in release repository..."
mkdir -p "${RELEASE_REPO}/Sources"
mkdir -p "${RELEASE_REPO}/Packages"

# Copy XCFramework
echo "📦 Copying AizipFishCount.xcframework..."
rm -rf "${RELEASE_REPO}/Packages/AizipFishCount.xcframework"
cp -R "${XCFRAMEWORK}" "${RELEASE_REPO}/Packages/"
echo "✅ XCFramework copied ($(du -sh "${RELEASE_REPO}/Packages/AizipFishCount.xcframework" | cut -f1))"

# Copy Visualization source code
echo "📄 Copying Visualization source code..."
rm -rf "${RELEASE_REPO}/Sources/Visualization"
cp -R "${PROJECT_ROOT}/Sources/Visualization" "${RELEASE_REPO}/Sources/"
echo "✅ Visualization source copied"

# NOTE: opencv2 is NOT copied
# - Not embedded in xcframework (bridging headers incompatible with BUILD_LIBRARY_FOR_DISTRIBUTION)
# - All OpenCV calls disabled in framework
# - Partners can optionally add opencv2 if needed for Phase 1 calibration

# Copy CoreML models
echo "🧠 Copying CoreML models..."
MODELS_SOURCE="${PROJECT_ROOT}/AizipFishCountApp/FishCountModels"
MODELS_DEST="${RELEASE_REPO}/AizipFishCountApp/FishCountModels"

if [ ! -d "${MODELS_SOURCE}" ]; then
    echo "⚠️  Warning: Models not found at: ${MODELS_SOURCE}"
else
    rm -rf "${MODELS_DEST}"
    mkdir -p "$(dirname "${MODELS_DEST}")"
    cp -R "${MODELS_SOURCE}" "${MODELS_DEST}"
    echo "✅ CoreML models copied"
fi

# Verify example app exists
EXAMPLE_APP="${RELEASE_REPO}/AizipFishCountApp"
if [ ! -d "${EXAMPLE_APP}" ]; then
    echo "⚠️  Warning: Example app not found at: ${EXAMPLE_APP}"
else
    echo "✅ Example app verified"
fi

# Verify framework source was cleaned (should be done by build script)
FRAMEWORK_SOURCE="${RELEASE_REPO}/AizipFishCountApp/AizipFishCount"
if [ -d "${FRAMEWORK_SOURCE}" ]; then
    echo "⚠️  Warning: Framework source still exists (should be cleaned by build script)"
    echo "   Run build_aizipfishcount_xcframework.sh to clean it up"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Release package ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Release repository contents:"
echo "  📦 Packages/AizipFishCount.xcframework  (binary SDK)"
echo "  📄 Sources/Visualization/              (UI source code)"
echo "  📱 AizipFishCountApp/                  (example app)"
echo "  🧠 FishCountModels/                    (CoreML models)"
echo ""
echo "Notes:"
echo "  - opencv2 NOT included (optional, partners add if needed)"
echo "  - AizipFishCount/ source cleaned by build script (not distributed)"
echo ""
echo "Next steps:"
echo "  1. cd ${RELEASE_REPO}"
echo "  2. git add ."
echo "  3. git commit -m \"Release v1.2.1\""
echo "  4. git push"
echo ""
echo "Partners will receive the Aizip_softbank_fishcount_ipad repository."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
