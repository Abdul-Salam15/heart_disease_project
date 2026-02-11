from django.contrib import admin
from .models import Prediction, BlockchainRecord


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'age', 'sex', 'risk_level', 'consensus_probability', 'created_at']
    list_filter = ['risk_level', 'sex', 'created_at']
    search_fields = ['id', 'age']
    readonly_fields = ['created_at']
    
    fieldsets = (
        ('Patient Information', {
            'fields': ('age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 
                      'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina', 'oldpeak')
        }),
        ('Prediction Results', {
            'fields': ('logistic_probability', 'rf_probability', 'consensus_probability',
                      'prediction_result', 'risk_level')
        }),
        ('Metadata', {
            'fields': ('created_at',)
        }),
    )


@admin.register(BlockchainRecord)
class BlockchainRecordAdmin(admin.ModelAdmin):
    list_display = ['block_index', 'block_hash_short', 'prediction', 'created_at']
    list_filter = ['created_at']
    search_fields = ['block_index', 'block_hash']
    readonly_fields = ['created_at']
    
    def block_hash_short(self, obj):
        return f"{obj.block_hash[:16]}..."
    block_hash_short.short_description = 'Block Hash'
