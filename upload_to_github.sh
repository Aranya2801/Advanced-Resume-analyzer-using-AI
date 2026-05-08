#!/usr/bin/env bash
# ============================================================
#  GITHUB UPLOAD SCRIPT
#  Run this from inside the project folder to push everything
#  to your GitHub repo for the first time.
# ============================================================

set -e

REPO_URL="https://github.com/Aranya2801/Advanced-Resume-analyzer-using-AI.git"

echo "🚀 Advanced Resume Analyzer — GitHub Upload Script"
echo "====================================================="

# Check git is installed
if ! command -v git &> /dev/null; then
  echo "❌ git is not installed. Please install git first."
  exit 1
fi

echo ""
echo "📋 Step 1: Initialise git repository"
git init
git branch -M main

echo ""
echo "📋 Step 2: Configure remote"
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"

echo ""
echo "📋 Step 3: Stage all files"
git add .

echo ""
echo "📋 Step 4: Create initial commit"
git commit -m "🚀 feat: MIT-grade Advanced Resume Analyzer v2.0

Complete rewrite with:
- Claude AI (claude-sonnet-4) deep analysis engine
- ATS scoring & semantic keyword matching (Sentence-BERT)
- Action verb analysis & quantification scoring
- AI bullet point rewriter & cover letter generator
- Career path intelligence with salary estimates
- FastAPI REST backend (/analyse, /cover-letter, /improve-bullet)
- Streamlit dark UI with Plotly dashboards
- Docker + docker-compose deployment
- GitHub Actions CI/CD (lint → test → security → docker)
- 20+ pytest tests with coverage
- Sample dataset generator (3 resumes, 2 JDs)
- Comprehensive MIT-grade README"

echo ""
echo "📋 Step 5: Push to GitHub"
echo "⚠️  You may be prompted for your GitHub username and Personal Access Token."
echo "    Generate a token at: https://github.com/settings/tokens"
echo ""
git push -u origin main --force

echo ""
echo "✅ SUCCESS! Your project is now live at:"
echo "   $REPO_URL"
echo ""
echo "🎉 Next steps:"
echo "   1. Add ANTHROPIC_API_KEY to GitHub Secrets (Settings → Secrets → Actions)"
echo "   2. Enable GitHub Actions in the Actions tab"
echo "   3. Add a project description and topics on GitHub"
echo "   4. Star your own repo to boost visibility!"
