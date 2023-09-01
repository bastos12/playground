from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_squared_error, r2_score



class MetricsScorer:

    def __init__(self, target_type, dict_value):
        self.type_model = target_type
        self.metrics_calcul = self._get_metrics_by_model_type(dict_value)

    def _get_metrics_by_model_type(self, dict_value):
        if isinstance(dict_value, dict) and len(dict_value) > 2:
            if self.type_model == 'classification':
                metrics = {
                    'accuracy': round(accuracy_score(dict_value['y_test'], dict_value['y_pred_test']), 2),
                    'recall': round(recall_score(dict_value['y_test'], dict_value['y_pred_test'], pos_label='classe_1'), 2),
                    'f1-score': round(f1_score(dict_value['y_test'], dict_value['y_pred_test'], pos_label='classe_1'), 2),
                }
                return metrics
            elif self.type_model == 'regression':
                metrics = {
                    'r2_score': round(r2_score(dict_value['y_test'], dict_value['y_pred_test']), 2),
                    'mse': round(mean_squared_error(dict_value['y_test'], dict_value['y_pred_test']), 2),
                }
                return metrics
            else:
                return "Problème de configuration des metrics"
        else:
            return "Pas d'évaluation possible"