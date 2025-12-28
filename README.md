<div align="center">
  <h1>Gemma Feature Studio</h1>
  <p>Gemma-3 SAE Analysis and Steering System built with Gemma-Scope-2, SAE-Lens, and React.</p>

  [![Gemma-3](https://img.shields.io/badge/LLM-Gemma--3-mediumorchid.svg)](https://ai.google.dev/gemma/docs/core)
  [![Gemma-Scope-2](https://img.shields.io/badge/SAE-Gemma--Scope--2-blue.svg)](https://huggingface.co/google/gemma-scope-2)
  [![Neuronpedia](https://img.shields.io/badge/Feature_Analysis-Neuronpedia-cadetblue.svg)](https://www.neuronpedia.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
</div>

## Features

- **Feature Activation Analysis**: Visualize which SAE features activate on your input text
- **Token-level Visualization**: Color-coded tokens showing activation strength
- **Feature Heatmaps**: Interactive heatmaps using visx
- **Network Visualization**: Interactive React Flow graph of feature activations
- **Virtualized Feature Lists**: Efficiently browse 64k+ features with react-window
- **Steering Controls**: Adjust feature coefficients (-2 to +2) to steer model output
- **Streaming Generation**: Real-time token streaming via SSE
- **Comparison Panel**: Side-by-side baseline vs steered output with diff view
- **Experiment Management**: Save, load, export, and import steering experiments
- **Undo/Redo**: Full history for steering coefficient changes (Ctrl+Z/Ctrl+Shift+Z)

## Architecture

```
lm_feature_studio/
├── frontend/          # Next.js 14 + React + Tailwind CSS
│   └── src/
│       ├── app/       # App router pages
│       ├── components/# React components (TokenDisplay, FeatureHeatmap, etc.)
│       ├── hooks/     # Custom React hooks (useAnalysis, useSelectedToken)
│       ├── lib/       # API client and utilities
│       └── types/     # TypeScript types
│
├── backend/           # Python FastAPI + SAELens
│   └── app/
│       ├── api/       # API endpoints (/analyze, /generate)
│       ├── core/      # Configuration
│       ├── inference/ # Model loading, analysis, steering
│       └── schemas/   # Pydantic models
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU with ~16GB VRAM (for Gemma 3 4B)

## Quick Start

```python
python start.py
```

This will:
1. Check dependencies and prompt to install if needed
2. Start the backend API on http://localhost:8000
3. Start the frontend UI on http://localhost:3000
4. Stream logs from both services
5. Handle graceful shutdown with Ctrl+C

## Manual Setup

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

The API will be available at http://localhost:8000. View API docs at http://localhost:8000/docs.

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The app will be available at http://localhost:3000.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check API health and model status |
| `/api/load` | POST | Load model and SAE into memory |
| `/api/unload` | POST | Unload model to free GPU memory |
| `/api/analyze` | POST | Analyze prompt and extract feature activations |
| `/api/generate` | POST | Generate text with optional steering |
| `/api/generate/stream` | POST | Streaming generation via SSE |
| `/api/model/info` | GET | Get loaded model information |

## Configuration

Backend configuration via environment variables (or `backend/.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `google/gemma-3-4b-it` | Gemma 3 model to load |
| `SAE_RELEASE` | `gemma-scope-2-4b-pt` | Gemma Scope 2 SAE release |
| `SAE_ID` | `layer_12/width_16k/average_l0_77` | Specific SAE configuration |
| `TARGET_LAYER` | `12` | Model layer to hook for SAE |
| `DEVICE` | `cuda` | Device for inference |
| `TOP_K_FEATURES` | `50` | Default number of top features per token |

See [Gemma Scope 2 on HuggingFace](https://huggingface.co/google/gemma-scope-2) for available SAE configurations.

## Usage

1. Start both backend and frontend servers
2. Click "Load Model" to load Gemma-3 and the SAE (may take a minute)
3. Enter a prompt and click "Analyze"
4. Click on tokens to see their feature activations
5. View the heatmap for cross-token feature patterns

## Technology Stack

- **Frontend**: Next.js 14, React 18, Tailwind CSS, React Flow, visx, react-window, Zustand
- **Backend**: FastAPI, HuggingFace Transformers, SAELens, PyTorch
- **Model**: Gemma 3 (270M to 27B) with Gemma Scope 2 SAEs

## License

MIT
