# Generated migration file for heart_disease_app

from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('age', models.IntegerField()),
                ('sex', models.CharField(max_length=10)),
                ('chest_pain_type', models.CharField(max_length=20)),
                ('resting_bp', models.IntegerField()),
                ('cholesterol', models.IntegerField()),
                ('fasting_bs', models.BooleanField()),
                ('resting_ecg', models.CharField(max_length=20)),
                ('max_hr', models.IntegerField()),
                ('exercise_angina', models.BooleanField()),
                ('oldpeak', models.FloatField()),
                ('logistic_probability', models.FloatField()),
                ('rf_probability', models.FloatField()),
                ('consensus_probability', models.FloatField()),
                ('prediction_result', models.CharField(max_length=50)),
                ('risk_level', models.CharField(max_length=20)),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='BlockchainRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('block_index', models.IntegerField()),
                ('timestamp', models.FloatField()),
                ('data', models.TextField()),
                ('previous_hash', models.CharField(max_length=64)),
                ('block_hash', models.CharField(max_length=64)),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('prediction', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='blockchain', to='heart_disease_app.prediction')),
            ],
            options={
                'ordering': ['block_index'],
            },
        ),
    ]
