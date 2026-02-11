from django.db import models
from django.utils import timezone
import hashlib


class Prediction(models.Model):
    """Model to store heart disease predictions"""
    
    # Patient Information
    age = models.IntegerField()
    sex = models.CharField(max_length=10)
    chest_pain_type = models.CharField(max_length=20)
    resting_bp = models.IntegerField()
    cholesterol = models.IntegerField()
    fasting_bs = models.BooleanField()
    resting_ecg = models.CharField(max_length=20)
    max_hr = models.IntegerField()
    exercise_angina = models.BooleanField()
    oldpeak = models.FloatField()
    
    # Prediction Results
    logistic_probability = models.FloatField()
    rf_probability = models.FloatField()
    consensus_probability = models.FloatField()
    prediction_result = models.CharField(max_length=50)
    risk_level = models.CharField(max_length=20)  # High or Low
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Prediction {self.id} - {self.risk_level} Risk ({self.created_at.strftime('%Y-%m-%d %H:%M')})"


class BlockchainRecord(models.Model):
    """Model to store blockchain records for predictions"""
    
    prediction = models.OneToOneField(Prediction, on_delete=models.CASCADE, related_name='blockchain')
    
    block_index = models.IntegerField()
    timestamp = models.FloatField()
    data = models.TextField()  # JSON string of prediction data
    previous_hash = models.CharField(max_length=64)
    block_hash = models.CharField(max_length=64)
    
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['block_index']
    
    def __str__(self):
        return f"Block {self.block_index} - {self.block_hash[:16]}..."
    
    @staticmethod
    def create_hash(block_index, timestamp, data, previous_hash):
        """Create SHA-256 hash for a block"""
        block_string = f"{block_index}{timestamp}{data}{previous_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()
