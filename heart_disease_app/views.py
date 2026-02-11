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
        
        # Load test data and models for evaluation
        try:
            import joblib
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
            import io
            import base64
            from django.conf import settings
            
            # Load models and test data
            log_model = joblib.load(settings.MODELS_DIR / 'logistic_model.pkl')
            rf_model = joblib.load(settings.MODELS_DIR / 'rf_model.pkl')
            X_test = joblib.load(settings.MODELS_DIR / 'X_test.pkl')
            y_test = joblib.load(settings.MODELS_DIR / 'y_test.pkl')
            
            # Generate charts and metrics for Logistic Regression
            log_pred = log_model.predict(X_test)
            log_pred_proba = log_model.predict_proba(X_test)[:, 1]
            
            # Confusion Matrix - Logistic
            fig, ax = plt.subplots(figsize=(6, 5))
            cm_log = confusion_matrix(y_test, log_pred)
            sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Disease', 'Heart Disease'],
                       yticklabels=['No Disease', 'Heart Disease'])
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title('Confusion Matrix - Logistic Regression', fontsize=14)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            logistic_cm_img = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()
            
            # ROC Curve - Logistic
            fig, ax = plt.subplots(figsize=(6, 5))
            fpr, tpr, _ = roc_curve(y_test, log_pred_proba)
            roc_auc_log = roc_auc_score(y_test, log_pred_proba)
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_log:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve - Logistic Regression', fontsize=14)
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            logistic_roc_img = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()
            
            # Metrics - Logistic
            logistic_metrics = {
                'accuracy': round(accuracy_score(y_test, log_pred) * 100, 2),
                'precision': round(precision_score(y_test, log_pred) * 100, 2),
                'recall': round(recall_score(y_test, log_pred) * 100, 2),
                'f1_score': round(f1_score(y_test, log_pred) * 100, 2),
                'roc_auc': round(roc_auc_log, 4)
            }
            
            # Generate charts and metrics for Random Forest
            rf_pred = rf_model.predict(X_test)
            rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            
            # Confusion Matrix - Random Forest
            fig, ax = plt.subplots(figsize=(6, 5))
            cm_rf = confusion_matrix(y_test, rf_pred)
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax,
                       xticklabels=['No Disease', 'Heart Disease'],
                       yticklabels=['No Disease', 'Heart Disease'])
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title('Confusion Matrix - Random Forest', fontsize=14)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            rf_cm_img = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()
            
            # ROC Curve - Random Forest
            fig, ax = plt.subplots(figsize=(6, 5))
            fpr, tpr, _ = roc_curve(y_test, rf_pred_proba)
            roc_auc_rf = roc_auc_score(y_test, rf_pred_proba)
            ax.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc_rf:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve - Random Forest', fontsize=14)
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            rf_roc_img = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()
            
            # Metrics - Random Forest
            rf_metrics = {
                'accuracy': round(accuracy_score(y_test, rf_pred) * 100, 2),
                'precision': round(precision_score(y_test, rf_pred) * 100, 2),
                'recall': round(recall_score(y_test, rf_pred) * 100, 2),
                'f1_score': round(f1_score(y_test, rf_pred) * 100, 2),
                'roc_auc': round(roc_auc_rf, 4)
            }
            
        except Exception as e:
            print(f"Error generating evaluation metrics: {e}")
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
