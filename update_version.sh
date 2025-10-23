#!/bin/bash
# Update version across all files and rebuild package

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to update version in a file
update_version() {
    local file=$1
    local old_version=$2
    local new_version=$3
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/$old_version/$new_version/g" "$file"
    else
        # Linux
        sed -i "s/$old_version/$new_version/g" "$file"
    fi
}

echo -e "${YELLOW}=== Version Update Script ===${NC}\n"

# Get current version from setup.py
CURRENT_VERSION=$(grep -oP 'version="\K[^"]+' setup.py 2>/dev/null || grep 'version=' setup.py | cut -d'"' -f2)
echo -e "Current version: ${GREEN}${CURRENT_VERSION}${NC}"

# Prompt for new version
read -p "Enter new version (e.g., 1.0.1): " NEW_VERSION

if [ -z "$NEW_VERSION" ]; then
    echo -e "${RED}Error: Version cannot be empty${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Updating version from ${CURRENT_VERSION} to ${NEW_VERSION}...${NC}\n"

# Update version in all files
echo "Updating setup.py..."
update_version "setup.py" "version=\"${CURRENT_VERSION}\"" "version=\"${NEW_VERSION}\""

echo "Updating pyproject.toml..."
update_version "pyproject.toml" "version = \"${CURRENT_VERSION}\"" "version = \"${NEW_VERSION}\""

echo "Updating src/__init__.py..."
update_version "src/__init__.py" "__version__ = \"${CURRENT_VERSION}\"" "__version__ = \"${NEW_VERSION}\""

echo -e "\n${GREEN}✓ Version updated in all files${NC}\n"

# Ask if user wants to run tests
read -p "Run tests before building? (y/n): " RUN_TESTS
if [[ "$RUN_TESTS" =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Running tests...${NC}"
    make test || {
        echo -e "${RED}✗ Tests failed! Fix issues before proceeding.${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ All tests passed${NC}\n"
fi

# Ask if user wants to build
read -p "Build package now? (y/n): " BUILD_NOW
if [[ "$BUILD_NOW" =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Cleaning old builds...${NC}"
    make clean-all
    
    echo -e "\n${YELLOW}Building package...${NC}"
    make build || {
        echo -e "${RED}✗ Build failed!${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Package built successfully${NC}\n"
    
    # Ask about TestPyPI upload
    read -p "Upload to TestPyPI? (y/n): " UPLOAD_TEST
    if [[ "$UPLOAD_TEST" =~ ^[Yy]$ ]]; then
        echo -e "\n${YELLOW}Uploading to TestPyPI...${NC}"
        make upload-test
    fi
fi

# Commit changes
read -p "Commit version changes to git? (y/n): " COMMIT_GIT
if [[ "$COMMIT_GIT" =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Committing version changes...${NC}"
    git add setup.py pyproject.toml src/__init__.py
    git commit -m "Bump version to ${NEW_VERSION}"
    echo -e "${GREEN}✓ Changes committed${NC}\n"
    
    # Tag release
    read -p "Create git tag v${NEW_VERSION}? (y/n): " CREATE_TAG
    if [[ "$CREATE_TAG" =~ ^[Yy]$ ]]; then
        git tag -a "v${NEW_VERSION}" -m "Release v${NEW_VERSION}"
        echo -e "${GREEN}✓ Tag v${NEW_VERSION} created${NC}\n"
        
        read -p "Push changes and tag to remote? (y/n): " PUSH_GIT
        if [[ "$PUSH_GIT" =~ ^[Yy]$ ]]; then
            git push && git push origin "v${NEW_VERSION}"
            echo -e "${GREEN}✓ Pushed to remote${NC}\n"
        fi
    fi
fi

echo -e "${GREEN}=== Update Complete! ===${NC}\n"
echo "Next steps:"
echo "  - Review changes: git status"
echo "  - Upload to PyPI: make upload"
echo "  - Install updated package: pip install --upgrade ctx-bandits-mcmc"
