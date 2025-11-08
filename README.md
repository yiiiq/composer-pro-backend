# Compose Pro Backend

A FastAPI-based backend service for serving machine learning models.

## Project Structure

```
compose-pro-backend/
├── Dockerfile            # Docker configuration
├── .dockerignore        # Docker ignore rules
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
└── app/                # Application code
    ├── __init__.py
    ├── main.py         # FastAPI application entry point
    ├── config.py       # Configuration settings
    ├── models/         # ML models directory
    │   ├── __init__.py
    │   ├── model_loader.py   # Model loading utilities
    │   ├── inference.py      # Inference logic
    │   └── saved_models/     # Directory for trained model files
    └── routers/        # API route handlers
        ├── __init__.py
        ├── health.py   # Health check endpoints
        └── predictions.py    # ML prediction endpoints
```

## Features

- ✅ FastAPI framework with automatic API documentation
- ✅ Modular router structure for organized endpoints
- ✅ ML model management (loading, inference)
- ✅ CORS middleware for cross-origin requests
- ✅ Environment-based configuration
- ✅ Health check endpoints
- ✅ Pydantic models for request/response validation

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Run the application

**Local Development:**
```bash
# Development mode with auto-reload
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Using Docker:**
```bash
# Build the Docker image
docker build -t compose-pro-backend .

# Run the container
docker run -p 8080:8080 compose-pro-backend

# Or with custom port
docker run -p 8000:8000 -e PORT=8000 compose-pro-backend
```

## API Endpoints

### Root
- `GET /` - Welcome message and API information

### Health
- `GET /api/health` - Service health check

### Predictions
- `POST /api/predict` - Make predictions using ML models
- `GET /api/models` - List available models

### API Documentation
- `GET /docs` - Interactive Swagger UI documentation
- `GET /redoc` - ReDoc documentation

## Adding Your ML Models

### 1. Save your trained model

Place your trained model files in `app/models/saved_models/`:

```bash
# Example: Save a scikit-learn model
import joblib
joblib.dump(model, 'app/models/saved_models/my_model.pkl')
```

### 2. Implement model loading

Update `app/models/model_loader.py` to load your specific model type:

```python
def load_model(self, model_name: str, model_path: Optional[str] = None):
    import joblib  # or torch, tensorflow, etc.
    model = joblib.load(model_path)
    self._models[model_name] = model
    return model
```

### 3. Implement inference logic

Update `app/models/inference.py` with your preprocessing and prediction logic:

```python
def predict(self, input_data: Any) -> Any:
    processed_data = self.preprocess(input_data)
    predictions = self.model.predict(processed_data)
    return predictions
```

### 4. Use in API endpoints

Update `app/routers/predictions.py` to use your models:

```python
from app.models.model_loader import model_loader
from app.models.inference import ModelInference

model = model_loader.load_model("my_model")
inference = ModelInference(model)
results = inference.run_inference(request.data)
```

## Development

### Run tests
```bash
pytest
```

### Format code
```bash
black .
```

### Type checking
```bash
mypy .
```

## Deployment

### Using Docker

The project includes a `Dockerfile` ready for deployment:

```bash
# Build the image
docker build -t compose-pro-backend .

# Run the container
docker run -p 8080:8080 compose-pro-backend

# Run with environment variables
docker run -p 8080:8080 -e PORT=8080 --env-file .env compose-pro-backend
```

### Docker Compose (optional)

Create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PORT=8080
    volumes:
      - ./app/models/saved_models:/app/app/models/saved_models
```

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

MIT License
