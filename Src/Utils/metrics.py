"""
Collecte et gestion des métriques de l'application.
"""

import time
import functools
from typing import Dict, Callable, Any
from collections import defaultdict
from datetime import datetime
from threading import Lock


class MetricsCollector:
    """Collecteur de métriques pour l'application."""
    
    def __init__(self):
        self._metrics: Dict[str, Any] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "errors": 0,
            "last_updated": None
        })
        self._lock = Lock()
    
    def record_execution(
        self,
        metric_name: str,
        execution_time: float,
        success: bool = True
    ):
        """
        Enregistre l'exécution d'une opération.
        
        Args:
            metric_name: Nom de la métrique
            execution_time: Temps d'exécution en secondes
            success: Si l'opération a réussi
        """
        with self._lock:
            metric = self._metrics[metric_name]
            metric["count"] += 1
            metric["total_time"] += execution_time
            metric["avg_time"] = metric["total_time"] / metric["count"]
            metric["min_time"] = min(metric["min_time"], execution_time)
            metric["max_time"] = max(metric["max_time"], execution_time)
            metric["last_updated"] = datetime.now().isoformat()
            
            if not success:
                metric["errors"] += 1
    
    def increment_counter(self, metric_name: str, value: int = 1):
        """
        Incrémente un compteur.
        
        Args:
            metric_name: Nom de la métrique
            value: Valeur à ajouter
        """
        with self._lock:
            if "value" not in self._metrics[metric_name]:
                self._metrics[metric_name]["value"] = 0
            self._metrics[metric_name]["value"] += value
            self._metrics[metric_name]["last_updated"] = datetime.now().isoformat()
    
    def set_gauge(self, metric_name: str, value: float):
        """
        Définit la valeur d'une jauge.
        
        Args:
            metric_name: Nom de la métrique
            value: Valeur à définir
        """
        with self._lock:
            self._metrics[metric_name]["value"] = value
            self._metrics[metric_name]["last_updated"] = datetime.now().isoformat()
    
    def get_metric(self, metric_name: str) -> Dict[str, Any]:
        """
        Récupère une métrique spécifique.
        
        Args:
            metric_name: Nom de la métrique
        
        Returns:
            Dictionnaire contenant les données de la métrique
        """
        with self._lock:
            return dict(self._metrics.get(metric_name, {}))
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Récupère toutes les métriques.
        
        Returns:
            Dictionnaire de toutes les métriques
        """
        with self._lock:
            return {name: dict(data) for name, data in self._metrics.items()}
    
    def reset_metric(self, metric_name: str):
        """Réinitialise une métrique."""
        with self._lock:
            if metric_name in self._metrics:
                del self._metrics[metric_name]
    
    def reset_all(self):
        """Réinitialise toutes les métriques."""
        with self._lock:
            self._metrics.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé des métriques.
        
        Returns:
            Résumé des métriques principales
        """
        metrics = self.get_all_metrics()
        total_calls = sum(m.get("count", 0) for m in metrics.values())
        total_errors = sum(m.get("errors", 0) for m in metrics.values())
        
        return {
            "total_metrics": len(metrics),
            "total_calls": total_calls,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_calls * 100) if total_calls > 0 else 0,
            "metrics": metrics
        }


# Instance globale du collecteur
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Retourne l'instance globale du collecteur de métriques."""
    return _global_collector


def track_time(metric_name: str = None):
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction.
    
    Args:
        metric_name: Nom de la métrique (utilise le nom de la fonction par défaut)
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                execution_time = time.time() - start_time
                _global_collector.record_execution(name, execution_time, success)
        
        return wrapper
    return decorator


def count_calls(metric_name: str = None):
    """
    Décorateur pour compter le nombre d'appels à une fonction.
    
    Args:
        metric_name: Nom de la métrique (utilise le nom de la fonction par défaut)
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}.calls"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _global_collector.increment_counter(name)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator