#!/bin/bash

# Heart Disease Prediction System - Setup Script
# This script helps set up the Django project

echo "🏥 Heart Disease Prediction System - Setup"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Dependencies installed successfully!"
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p models_data
mkdir -p data
mkdir -p media
mkdir -p staticfiles

echo "✅ Directories created!"
echo ""

# Database setup
echo "🗄️  Setting up database..."
python manage.py makemigrations
python manage.py migrate

echo ""
echo "✅ Database setup complete!"
echo ""

# Collect static files
echo "📦 Collecting static files..."
python manage.py collectstatic --noinput

echo ""
echo "✅ Static files collected!"
echo ""

# Instructions
echo "=========================================="
echo "🎉 Setup Complete!"
echo "=========================================="
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Copy your dataset to the data/ directory:"
echo "   cp /path/to/heart_disease_combined.csv data/"
echo ""
echo "2. Train the ML models:"
echo "   python train_model.py"
echo "   (Make sure model files are saved to models_data/ directory)"
echo ""
echo "3. Create a superuser (optional):"
echo "   python manage.py createsuperuser"
echo ""
echo "4. Run the development server:"
echo "   python manage.py runserver"
echo ""
echo "5. Open your browser to:"
echo "   http://127.0.0.1:8000"
echo ""
echo "=========================================="
echo "📚 Useful Commands:"
echo "=========================================="
echo ""
echo "Start server:        python manage.py runserver"
echo "Create superuser:    python manage.py createsuperuser"
echo "Make migrations:     python manage.py makemigrations"
echo "Apply migrations:    python manage.py migrate"
echo "Access admin:        http://127.0.0.1:8000/admin"
echo ""
echo "=========================================="
