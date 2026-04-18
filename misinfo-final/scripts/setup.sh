#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# AI Against Misinformation — One-shot setup script
# Run from project root: bash scripts/setup.sh
# ─────────────────────────────────────────────────────────────────
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

log() { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

log "🚀 AI Against Misinformation — Setup"
echo "────────────────────────────────────────"

# 1. Python check
python3 --version >/dev/null 2>&1 || err "Python 3.9+ required."
PYTHON_VER=$(python3 -c "import sys; print(sys.version_info.minor)")
[ "$PYTHON_VER" -ge 9 ] || err "Python 3.9+ required. Found 3.$PYTHON_VER"
log "✅ Python 3.$PYTHON_VER"

# 2. Create venv
if [ ! -d "backend/venv" ]; then
    log "Creating virtual environment…"
    python3 -m venv backend/venv
fi
source backend/venv/bin/activate

# 3. Install dependencies
log "Installing Python dependencies (this may take a few minutes)…"
pip install --upgrade pip -q
pip install -r backend/requirements.txt -q
log "✅ Dependencies installed"

# 4. .env check
if [ ! -f "backend/.env" ]; then
    cp backend/.env.example backend/.env
    warn "Created backend/.env from example. Please fill in your API keys."
fi
log "✅ .env file present"

# 5. Initialise Qdrant collections
log "Initialising Qdrant collections…"
cd backend
python ../qdrant/init_collections.py && log "✅ Qdrant collections created"
cd ..

# 6. Seed data
read -p "Seed Qdrant with demo data? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd backend && python ../qdrant/seed_data.py && cd ..
    log "✅ Seed data loaded"
fi

echo ""
echo "────────────────────────────────────────"
log "✅ Setup complete!"
echo ""
echo "  Start API:   cd backend && uvicorn app.main:app --reload --port 8000"
echo "  API Docs:    http://localhost:8000/docs"
echo "  Health:      http://localhost:8000/health"
echo "────────────────────────────────────────"
