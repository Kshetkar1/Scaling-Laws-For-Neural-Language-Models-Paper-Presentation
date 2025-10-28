#!/bin/bash

echo "=========================================="
echo "GitHub Push Script"
echo "=========================================="
echo ""
echo "This script will push your repository to GitHub."
echo ""
echo "BEFORE RUNNING THIS SCRIPT:"
echo "1. Go to https://github.com/new"
echo "2. Create a new repository named: Scaling-Laws-For-Neural-Language-Models-Paper-Presentation"
echo "3. Do NOT initialize with README"
echo "4. Click 'Create repository'"
echo ""
read -p "Have you created the GitHub repository? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please create the repository first, then run this script again."
    exit 1
fi

echo ""
echo "Setting up remote..."
git remote add origin https://github.com/Kshetkar1/Scaling-Laws-For-Neural-Language-Models-Paper-Presentation.git 2>/dev/null || echo "Remote already exists"

echo ""
echo "Renaming branch to main..."
git branch -M main

echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "=========================================="
echo "✅ Done! Your repository should now be on GitHub at:"
echo "https://github.com/Kshetkar1/Scaling-Laws-For-Neural-Language-Models-Paper-Presentation"
echo "=========================================="
