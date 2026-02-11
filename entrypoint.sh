#!/usr/bin/env bash
set -e

# run migrations and collectstatic before starting the app
echo "Running migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Gunicorn..."
exec gunicorn heart_disease_project.wsgi:application --bind 0.0.0.0:8000 --workers 3
