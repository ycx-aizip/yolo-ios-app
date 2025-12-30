#!/bin/bash
#
# Flatten OpenCV frameworks and build xcframework
#
# This script converts deep bundle frameworks (with Versions/) to shallow bundles
# then creates a proper xcframework for iOS
#

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Flattening OpenCV frameworks and building xcframework...${NC}\n"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR/.."
OPENCV_DIR="$REPO_ROOT/Packages/opencv2"

# Temporary directory for flattened frameworks
TEMP_DIR="$OPENCV_DIR/temp_flat"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Function to flatten a framework
flatten_framework() {
    local source_framework="$1"
    local dest_framework="$2"

    echo -e "${YELLOW}Flattening: $(basename $source_framework)${NC}"

    # Create destination framework directory
    mkdir -p "$dest_framework"

    # Check if this is a deep bundle (has Versions directory)
    if [ -d "$source_framework/Versions" ]; then
        echo "  - Deep bundle detected, flattening..."

        # Copy the actual files from Versions/Current (or Versions/A)
        if [ -L "$source_framework/Versions/Current" ]; then
            # Follow symlink to actual version
            local version_dir=$(readlink "$source_framework/Versions/Current")
            cp -R "$source_framework/Versions/$version_dir"/* "$dest_framework/"
        elif [ -d "$source_framework/Versions/A" ]; then
            cp -R "$source_framework/Versions/A"/* "$dest_framework/"
        else
            echo -e "${RED}Error: Unknown Versions structure${NC}"
            exit 1
        fi
    else
        echo "  - Already flat, copying..."
        cp -R "$source_framework"/* "$dest_framework/"
    fi

    # Verify the binary exists
    local binary_name=$(basename "$dest_framework" .framework)
    if [ ! -f "$dest_framework/$binary_name" ]; then
        echo -e "${RED}Error: Binary not found at $dest_framework/$binary_name${NC}"
        exit 1
    fi

    # Create Info.plist if it doesn't exist
    if [ ! -f "$dest_framework/Info.plist" ]; then
        echo "  - Creating Info.plist..."
        cat > "$dest_framework/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$binary_name</string>
    <key>CFBundleIdentifier</key>
    <string>org.opencv.$binary_name</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$binary_name</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>4.10.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>iPhoneOS</string>
        <string>iPhoneSimulator</string>
    </array>
    <key>MinimumOSVersion</key>
    <string>14.0</string>
</dict>
</plist>
EOF
    fi

    echo "  - ✅ Flattened successfully"
}

# Flatten iOS device framework
if [ -d "$OPENCV_DIR/ios/opencv2.framework" ]; then
    flatten_framework "$OPENCV_DIR/ios/opencv2.framework" "$TEMP_DIR/ios/opencv2.framework"
else
    echo -e "${RED}Error: iOS framework not found${NC}"
    exit 1
fi

# Flatten iOS simulator framework
if [ -d "$OPENCV_DIR/ios-simulator/opencv2.framework" ]; then
    flatten_framework "$OPENCV_DIR/ios-simulator/opencv2.framework" "$TEMP_DIR/ios-simulator/opencv2.framework"
else
    echo -e "${RED}Error: Simulator framework not found${NC}"
    exit 1
fi

# Remove old xcframework
if [ -d "$OPENCV_DIR/opencv2.xcframework" ]; then
    echo -e "\n${YELLOW}Removing old xcframework...${NC}"
    rm -rf "$OPENCV_DIR/opencv2.xcframework"
fi

# Create xcframework from flattened frameworks
echo -e "\n${YELLOW}Creating xcframework...${NC}"
xcodebuild -create-xcframework \
    -framework "$TEMP_DIR/ios/opencv2.framework" \
    -framework "$TEMP_DIR/ios-simulator/opencv2.framework" \
    -output "$OPENCV_DIR/opencv2.xcframework"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Successfully created opencv2.xcframework (flattened)${NC}"

    # Clean up temp directory
    rm -rf "$TEMP_DIR"

    # Verify structure
    echo -e "\n${YELLOW}Verifying framework structure:${NC}"
    if [ -d "$OPENCV_DIR/opencv2.xcframework/ios-arm64_x86_64-simulator/opencv2.framework/Versions" ]; then
        echo -e "${RED}❌ WARNING: Framework still has Versions directory!${NC}"
    else
        echo -e "${GREEN}✅ Framework is properly flattened${NC}"
    fi

    ls -lh "$OPENCV_DIR/opencv2.xcframework/ios-arm64_x86_64-simulator/opencv2.framework/" | head -10

    echo -e "\n${GREEN}Done! You can now use opencv2.xcframework with 'Embed & Sign'${NC}"
else
    echo -e "${RED}❌ Failed to create xcframework${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi
