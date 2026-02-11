from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
import time
import json
from .models import Prediction, BlockchainRecord
from .utils import ml_manager, DataValidator, DataPreprocessor, BlockchainManager


def home(request):
    """Home page view"""
    return render(request, 'home.html')


def predict_view(request):
    """Prediction page view"""
    if request.method == 'POST':
        try:
            # Extract form data
            input_data = {
                'age': int(request.POST.get('age')),
                'sex': request.POST.get('sex'),
                'chest_pain_type': request.POST.get('chest_pain_type'),
                'resting_bp': int(request.POST.get('resting_bp')),
                'cholesterol': int(request.POST.get('cholesterol')),
                'fasting_bs': request.POST.get('fasting_bs') == 'Yes',
                'resting_ecg': request.POST.get('resting_ecg'),
                'max_hr': int(request.POST.get('max_hr')),
                'exercise_angina': request.POST.get('exercise_angina') == 'Yes',
                'oldpeak': float(request.POST.get('oldpeak')),
            }
            
            # Validate input
            is_valid, errors = DataValidator.validate_input(input_data)
            if not is_valid:
                for error in errors:
                    messages.error(request, error)
                return redirect('predict')
            
            # Clean input
            input_data = DataValidator.clean_input(input_data)
            
            # Make prediction using ml_manager
            result = ml_manager.predict(input_data)
            
            # Save to database
            prediction = Prediction.objects.create(
                age=input_data['age'],
                sex=input_data['sex'],
                chest_pain_type=input_data['chest_pain_type'],
                resting_bp=input_data['resting_bp'],
                cholesterol=input_data['cholesterol'],
                fasting_bs=input_data['fasting_bs'],
                resting_ecg=input_data['resting_ecg'],
                max_hr=input_data['max_hr'],
                exercise_angina=input_data['exercise_angina'],
                oldpeak=input_data['oldpeak'],
                logistic_probability=result['logistic_prob'] * 100,
                rf_probability=result['rf_prob'] * 100,
                consensus_probability=result['consensus_prob'] * 100,
                prediction_result=result['prediction_result'],
                risk_level=result['risk_level']
            )
            
            # Create blockchain record
            previous_block = BlockchainRecord.objects.last()
            block_index = (previous_block.block_index + 1) if previous_block else 1
            previous_hash = previous_block.block_hash if previous_block else '0'
            timestamp = time.time()
            
            block_data = {
                'patient_age': input_data['age'],
                'sex': input_data['sex'],
                'logistic_prob': round(result['logistic_prob'] * 100, 2),
                'rf_prob': round(result['rf_prob'] * 100, 2),
                'consensus_prob': round(result['consensus_prob'] * 100, 2),
                'prediction': result['risk_level'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            block_hash = BlockchainRecord.create_hash(
                block_index, timestamp, json.dumps(block_data), previous_hash
            )
            
            BlockchainRecord.objects.create(
                prediction=prediction,
                block_index=block_index,
                timestamp=timestamp,
                data=json.dumps(block_data),
                previous_hash=previous_hash,
                block_hash=block_hash
            )
            
            # Redirect to results with prediction ID
            return redirect('results', prediction_id=prediction.id)
            
        except Exception as e:
            messages.error(request, f'Error making prediction: {str(e)}')
            return redirect('predict')
    
    return render(request, 'predict.html')


def results_view(request, prediction_id):
    """Results page view"""
    try:
        prediction = Prediction.objects.get(id=prediction_id)
        blockchain = prediction.blockchain
        
        # Use precomputed evaluation artifacts from ml_manager to avoid slow
        # per-request computation. These are prepared at import/startup if
        # test data is available.
        try:
            logistic_cm_img = getattr(ml_manager, 'logistic_confusion_matrix', None)
            logistic_roc_img = getattr(ml_manager, 'logistic_roc_curve', None)
            logistic_metrics = getattr(ml_manager, 'logistic_metrics', {}) or {}
            rf_cm_img = getattr(ml_manager, 'rf_confusion_matrix', None)
            rf_roc_img = getattr(ml_manager, 'rf_roc_curve', None)
            rf_metrics = getattr(ml_manager, 'rf_metrics', {}) or {}
        except Exception as e:
            print(f"Error fetching precomputed evaluation artifacts: {e}")
            logistic_cm_img = None
            logistic_roc_img = None
            rf_cm_img = None
            rf_roc_img = None
            logistic_metrics = {}
            rf_metrics = {}
        
        context = {
            'prediction': prediction,
            'blockchain': blockchain,
            'logistic_confusion_matrix': logistic_cm_img,
            'logistic_roc_curve': logistic_roc_img,
            'logistic_metrics': logistic_metrics,
            'rf_confusion_matrix': rf_cm_img,
            'rf_roc_curve': rf_roc_img,
            'rf_metrics': rf_metrics,
        }
        return render(request, 'results.html', context)
    except Prediction.DoesNotExist:
        messages.error(request, 'Prediction not found')
        return redirect('predict')


def blockchain_view(request):
    """Blockchain records page view"""
    blocks = BlockchainRecord.objects.all().select_related('prediction')
    
    context = {
        'blocks': blocks,
        'total_blocks': blocks.count()
    }
    return render(request, 'blockchain.html', context)


def history_view(request):
    """Prediction history page view"""
    predictions = Prediction.objects.all()
    high_count = predictions.filter(risk_level='High').count()
    low_count = predictions.count() - high_count

    context = {
        'predictions': predictions,
        'total_predictions': predictions.count(),
        'high_risk_count': high_count,
        'low_risk_count': low_count,
    }
    return render(request, 'history.html', context)
