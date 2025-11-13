# FastAPI Backend

A clean and well-structured FastAPI application following best practices.

## Project Structure

```
backend/
├── main.py                  # Application entry point (run with python main.py)
├── app/
│   ├── __init__.py
│   ├── main.py              # Main FastAPI application
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── api.py       # API router aggregation
│   │       └── endpoints/   # API endpoint routers
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Application settings
│   │   └── security.py      # Security utilities
│   ├── schemas/             # Pydantic models/schemas
│   ├── models/              # Database models
│   ├── dependencies.py      # Reusable dependencies
│   └── utils/               # Utility functions
├── tests/                   # Test files
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables example
└── README.md               # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

4. Run the application:

**Recommended (using main.py):**
```bash
python main.py
```

**Alternative methods:**
```bash
# Using uvicorn directly
uvicorn app.main:app --reload

# Using FastAPI CLI
fastapi dev app/main.py
```

## Development

- API Documentation: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

## Testing

Run tests with pytest:
```bash
pytest
```

## Code Quality

Format code:
```bash
black .
isort .
```

Check code quality:
```bash
flake8 .
mypy .
```

