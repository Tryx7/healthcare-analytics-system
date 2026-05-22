
healthcare-analytics/
│
├── 📄 api.py                    # Standalone Flask API (deploy this)
├── 📄 app.py                    # Alternative Flask app
├── 📄 main.py                   # FastAPI application
│
├── 📁 models/
│   ├── model_v1.0.0.pkl        # Pre-trained model
│   └── model_latest.pkl        # Latest model (used by API)
│
├── 📄 data_cleaning.py          # Data cleaning module
├── 📄 database.py               # PostgreSQL database module
├── 📄 ml_model.py               # Machine learning module
├── 📄 scheduler.py              # Weekly retraining scheduler
├── 📄 initialize.py             # System initialization script
│
├── 📄 healthcare_dataset.csv    # Raw dataset
├── 📄 cleaned_healthcare_data.csv # Cleaned dataset
│
├── 📄 requirements.txt          # Python dependencies
├── 📄 requirements-flask.txt    # Flask-only dependencies
│
├── 📄 Dockerfile                # Docker configuration
├── 📄 docker-compose.yml        # Docker Compose setup
├── 📄 vercel.json               # Vercel deployment config
│
├── 📄 .env.example              # Environment variables template
├── 📄 .gitignore                # Git ignore rules
│
├── 📄 README.md                 # Project documentation
├── 📄 test_api.py               # API testing script
│
└── 📁 .github/
    └── 📁 workflows/
        └── ci-cd.yml            # GitHub Actions CI/CD
