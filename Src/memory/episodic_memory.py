import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types d'événements possibles"""
    ACTION = "action"
    OBSERVATION = "observation"
    INTERACTION = "interaction"
    DECISION = "decision"
    LEARNING = "learning"
    ERROR = "error"
    SUCCESS = "success"
    COMMUNICATION = "communication"


class Episode:
    """
    Représente un épisode de mémoire
    Un épisode est une séquence cohérente d'événements liés
    """
    
    def __init__(
        self,
        episode_id: str,
        events: Optional[List[Dict[str, Any]]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.episode_id = episode_id
        self.events = events or []
        self.start_time = start_time or datetime.now()
        self.end_time = end_time
        self.metadata = metadata or {}
        self.tags = set()
        self.summary = None
        self.importance = 0.5
        self.emotional_valence = 0.0  # -1 (négatif) à +1 (positif)
        self.outcome = None  # success, failure, neutral
        
    def add_event(
        self,
        event_type: EventType,
        content: Any,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Ajouter un événement à l'épisode"""
        event = {
            "type": event_type.value if isinstance(event_type, EventType) else event_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "metadata": metadata or {},
            "sequence_num": len(self.events)
        }
        
        self.events.append(event)
        self.end_time = datetime.now()
        
        # Mettre à jour la valence émotionnelle
        self._update_emotional_valence(event)
    
    def _update_emotional_valence(self, event: Dict[str, Any]):
        """Mettre à jour la valence émotionnelle basée sur les événements"""
        event_type = event["type"]
        
        # Ajustements basés sur le type d'événement
        if event_type == EventType.SUCCESS.value:
            self.emotional_valence = min(1.0, self.emotional_valence + 0.2)
        elif event_type == EventType.ERROR.value:
            self.emotional_valence = max(-1.0, self.emotional_valence - 0.2)
        
        # Ajustements basés sur les métadonnées
        if "sentiment" in event.get("metadata", {}):
            sentiment = event["metadata"]["sentiment"]
            if sentiment == "positive":
                self.emotional_valence = min(1.0, self.emotional_valence + 0.1)
            elif sentiment == "negative":
                self.emotional_valence = max(-1.0, self.emotional_valence - 0.1)
    
    def add_tag(self, tag: str):
        """Ajouter un tag à l'épisode"""
        self.tags.add(tag.lower())
    
    def add_tags(self, tags: List[str]):
        """Ajouter plusieurs tags"""
        for tag in tags:
            self.add_tag(tag)
    
    def remove_tag(self, tag: str):
        """Retirer un tag"""
        self.tags.discard(tag.lower())
    
    def set_summary(self, summary: str):
        """Définir un résumé de l'épisode"""
        self.summary = summary
    
    def set_outcome(self, outcome: str):
        """Définir le résultat de l'épisode"""
        if outcome in ["success", "failure", "neutral"]:
            self.outcome = outcome
    
    def duration_seconds(self) -> float:
        """Durée de l'épisode en secondes"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def get_event_types(self) -> Dict[str, int]:
        """Obtenir la distribution des types d'événements"""
        type_counts = {}
        for event in self.events:
            event_type = event["type"]
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        return type_counts
    
    def get_involved_agents(self) -> set:
        """Obtenir tous les agents impliqués dans l'épisode"""
        agents = set()
        for event in self.events:
            agent_id = event.get("agent_id")
            if agent_id:
                agents.add(agent_id)
        return agents
    
    def get_key_moments(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        Identifier les moments clés de l'épisode
        
        Args:
            n: Nombre de moments clés à retourner
            
        Returns:
            Liste des événements les plus importants
        """
        # Prioriser certains types d'événements
        priority_types = [
            EventType.DECISION.value,
            EventType.SUCCESS.value,
            EventType.ERROR.value,
            EventType.LEARNING.value
        ]
        
        key_events = []
        
        # Ajouter les événements prioritaires
        for event in self.events:
            if event["type"] in priority_types:
                key_events.append(event)
        
        # Si pas assez, ajouter les premiers et derniers
        if len(key_events) < n and self.events:
            if self.events[0] not in key_events:
                key_events.insert(0, self.events[0])
            if self.events[-1] not in key_events:
                key_events.append(self.events[-1])
        
        return key_events[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        return {
            "episode_id": self.episode_id,
            "events": self.events,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "summary": self.summary,
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "outcome": self.outcome,
            "duration_seconds": self.duration_seconds(),
            "event_count": len(self.events)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        """Créer depuis un dictionnaire"""
        episode = cls(
            episode_id=data["episode_id"],
            events=data["events"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            metadata=data.get("metadata", {})
        )
        episode.tags = set(data.get("tags", []))
        episode.summary = data.get("summary")
        episode.importance = data.get("importance", 0.5)
        episode.emotional_valence = data.get("emotional_valence", 0.0)
        episode.outcome = data.get("outcome")
        return episode
    
    def __len__(self) -> int:
        """Nombre d'événements dans l'épisode"""
        return len(self.events)
    
    def __repr__(self) -> str:
        return (
            f"Episode({self.episode_id}, "
            f"events={len(self.events)}, "
            f"duration={self.duration_seconds():.1f}s, "
            f"importance={self.importance:.2f})"
        )


class EpisodicMemory:
    """
    Mémoire épisodique pour stocker des séquences d'événements
    Similaire à la mémoire autobiographique humaine
    Stocke des expériences complètes avec contexte temporel et relationnel
    """
    
    def __init__(
        self,
        max_episodes: int = 1000,
        auto_summarize: bool = True,
        persist_path: Optional[str] = None,
        importance_decay: float = 0.95,
        consolidation_interval: int = 50
    ):
        """
        Args:
            max_episodes: Nombre maximum d'épisodes à conserver
            auto_summarize: Créer automatiquement des résumés
            persist_path: Chemin pour la persistance
            importance_decay: Facteur de décroissance de l'importance (0-1)
            consolidation_interval: Intervalle d'épisodes avant consolidation
        """
        self.max_episodes = max_episodes
        self.auto_summarize = auto_summarize
        self.persist_path = persist_path
        self.importance_decay = importance_decay
        self.consolidation_interval = consolidation_interval
        
        # Stockage des épisodes
        self.episodes: Dict[str, Episode] = {}
        self.current_episode: Optional[Episode] = None
        
        # Index pour recherche rapide
        self._episodes_by_tag: Dict[str, List[str]] = {}
        self._episodes_by_agent: Dict[str, List[str]] = {}
        self._episodes_by_outcome: Dict[str, List[str]] = {}
        self._temporal_index: List[Tuple[datetime, str]] = []  # (timestamp, episode_id)
        
        # Callbacks
        self._on_episode_start: List[Callable] = []
        self._on_episode_end: List[Callable] = []
        
        # Statistiques
        self.stats = {
            "total_episodes": 0,
            "total_events": 0,
            "active_episode": None,
            "episodes_pruned": 0,
            "consolidations": 0,
            "average_duration": 0.0,
            "average_events_per_episode": 0.0
        }
        
        # Charger depuis le disque si disponible
        if persist_path and Path(persist_path).exists():
            self.load()
    
    def register_callback(self, event: str, callback: Callable):
        """
        Enregistrer un callback pour un événement
        
        Args:
            event: 'episode_start' ou 'episode_end'
            callback: Fonction à appeler
        """
        if event == "episode_start":
            self._on_episode_start.append(callback)
        elif event == "episode_end":
            self._on_episode_end.append(callback)
    
    def start_episode(
        self,
        episode_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_end_previous: bool = True
    ) -> str:
        """
        Commencer un nouvel épisode
        
        Args:
            episode_id: ID de l'épisode (généré si None)
            metadata: Métadonnées de l'épisode
            auto_end_previous: Terminer automatiquement l'épisode précédent
            
        Returns:
            ID de l'épisode
        """
        # Terminer l'épisode actuel si existant
        if self.current_episode and auto_end_previous:
            self.end_episode()
        
        # Générer un ID si nécessaire
        if episode_id is None:
            episode_id = f"ep_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        # Créer le nouvel épisode
        self.current_episode = Episode(
            episode_id=episode_id,
            start_time=datetime.now(),
            metadata=metadata
        )
        
        self.stats["active_episode"] = episode_id
        logger.info(f"Nouvel épisode commencé: {episode_id}")
        
        # Appeler les callbacks
        for callback in self._on_episode_start:
            try:
                callback(self.current_episode)
            except Exception as e:
                logger.error(f"Erreur dans callback episode_start: {e}")
        
        return episode_id
    
    def add_event(
        self,
        event_type: EventType,
        content: Any,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_start: bool = True
    ):
        """
        Ajouter un événement à l'épisode actuel
        
        Args:
            event_type: Type d'événement
            content: Contenu de l'événement
            agent_id: ID de l'agent concerné
            metadata: Métadonnées additionnelles
            auto_start: Créer automatiquement un épisode si nécessaire
        """
        if not self.current_episode:
            if auto_start:
                self.start_episode()
            else:
                logger.warning("Aucun épisode actif et auto_start=False")
                return
        
        self.current_episode.add_event(
            event_type=event_type,
            content=content,
            agent_id=agent_id,
            metadata=metadata
        )
        
        self.stats["total_events"] += 1
        
        logger.debug(
            f"Événement {event_type.value} ajouté à l'épisode "
            f"{self.current_episode.episode_id}"
        )
    
    def end_episode(
        self,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: Optional[float] = None,
        outcome: Optional[str] = None
    ) -> Optional[Episode]:
        """
        Terminer l'épisode actuel et le sauvegarder
        
        Args:
            summary: Résumé de l'épisode
            tags: Tags pour l'épisode
            importance: Score d'importance (0-1)
            outcome: Résultat (success, failure, neutral)
            
        Returns:
            L'épisode terminé ou None
        """
        if not self.current_episode:
            logger.warning("Aucun épisode actif à terminer")
            return None
        
        episode = self.current_episode
        
        # Ajouter les tags
        if tags:
            episode.add_tags(tags)
        
        # Définir l'importance
        if importance is not None:
            episode.importance = max(0.0, min(1.0, importance))
        else:
            # Calculer l'importance automatiquement
            episode.importance = self._calculate_importance(episode)
        
        # Définir le résultat
        if outcome:
            episode.set_outcome(outcome)
        
        # Créer un résumé si nécessaire
        if summary:
            episode.set_summary(summary)
        elif self.auto_summarize and len(episode.events) > 0:
            episode.set_summary(self._generate_auto_summary(episode))
        
        # Finaliser l'épisode
        episode.end_time = datetime.now()
        
        # Sauvegarder l'épisode
        self.episodes[episode.episode_id] = episode
        self.stats["total_episodes"] += 1
        
        # Mettre à jour les index
        self._update_indexes(episode)
        
        # Mettre à jour les statistiques
        self._update_statistics()
        
        # Appeler les callbacks
        for callback in self._on_episode_end:
            try:
                callback(episode)
            except Exception as e:
                logger.error(f"Erreur dans callback episode_end: {e}")
        
        logger.info(
            f"Épisode terminé: {episode.episode_id} "
            f"({len(episode.events)} événements, "
            f"{episode.duration_seconds():.1f}s, "
            f"importance={episode.importance:.2f})"
        )
        
        # Consolider si nécessaire
        if self.stats["total_episodes"] % self.consolidation_interval == 0:
            self.consolidate()
        
        # Gérer la limite de capacité
        if len(self.episodes) > self.max_episodes:
            self._prune_episodes()
        
        # Sauvegarder si chemin configuré
        if self.persist_path:
            self.save()
        
        ended_episode = self.current_episode
        self.current_episode = None
        self.stats["active_episode"] = None
        
        return ended_episode
    
    def _calculate_importance(self, episode: Episode) -> float:
        """
        Calculer automatiquement l'importance d'un épisode
        Basé sur plusieurs facteurs
        """
        importance = 0.5  # Base
        
        # Facteur 1: Nombre d'événements
        event_count = len(episode.events)
        if event_count > 10:
            importance += 0.1
        if event_count > 20:
            importance += 0.1
        
        # Facteur 2: Durée
        duration = episode.duration_seconds()
        if duration > 300:  # 5 minutes
            importance += 0.1
        
        # Facteur 3: Types d'événements importants
        event_types = episode.get_event_types()
        if EventType.LEARNING.value in event_types:
            importance += 0.2
        if EventType.ERROR.value in event_types:
            importance += 0.1
        if EventType.SUCCESS.value in event_types:
            importance += 0.15
        
        # Facteur 4: Valence émotionnelle extrême
        if abs(episode.emotional_valence) > 0.5:
            importance += 0.15
        
        # Facteur 5: Résultat
        if episode.outcome == "success":
            importance += 0.1
        elif episode.outcome == "failure":
            importance += 0.15  # Les échecs sont importants pour l'apprentissage
        
        return max(0.0, min(1.0, importance))
    
    def _generate_auto_summary(self, episode: Episode) -> str:
        """Générer automatiquement un résumé d'épisode"""
        num_events = len(episode.events)
        duration = episode.duration_seconds()
        
        # Extraire les types d'événements
        event_types = episode.get_event_types()
        
        # Agents impliqués
        agents = episode.get_involved_agents()
        
        # Construire le résumé
        summary = f"Épisode avec {num_events} événements sur {duration:.1f}s. "
        
        if agents:
            summary += f"Agents: {', '.join(list(agents)[:3])}. "
        
        if event_types:
            top_types = sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:3]
            summary += "Types: " + ", ".join(f"{t}({c})" for t, c in top_types) + ". "
        
        if episode.outcome:
            summary += f"Résultat: {episode.outcome}. "
        
        if episode.emotional_valence != 0:
            valence_desc = "positive" if episode.emotional_valence > 0 else "négative"
            summary += f"Valence {valence_desc} ({episode.emotional_valence:.2f})."
        
        return summary
    
    def _update_indexes(self, episode: Episode):
        """Mettre à jour les index de recherche"""
        episode_id = episode.episode_id
        
        # Index par tags
        for tag in episode.tags:
            if tag not in self._episodes_by_tag:
                self._episodes_by_tag[tag] = []
            self._episodes_by_tag[tag].append(episode_id)
        
        # Index par agent
        agents = episode.get_involved_agents()
        for agent_id in agents:
            if agent_id not in self._episodes_by_agent:
                self._episodes_by_agent[agent_id] = []
            self._episodes_by_agent[agent_id].append(episode_id)
        
        # Index par résultat
        if episode.outcome:
            if episode.outcome not in self._episodes_by_outcome:
                self._episodes_by_outcome[episode.outcome] = []
            self._episodes_by_outcome[episode.outcome].append(episode_id)
        
        # Index temporel
        self._temporal_index.append((episode.start_time, episode_id))
        self._temporal_index.sort(key=lambda x: x[0])
    
    def _update_statistics(self):
        """Mettre à jour les statistiques globales"""
        if not self.episodes:
            return
        
        total_duration = sum(ep.duration_seconds() for ep in self.episodes.values())
        total_events = sum(len(ep.events) for ep in self.episodes.values())
        
        self.stats["average_duration"] = total_duration / len(self.episodes)
        self.stats["average_events_per_episode"] = total_events / len(self.episodes)
    
    def consolidate(self):
        """
        Consolider la mémoire épisodique
        Appliquer la décroissance d'importance et fusionner les épisodes similaires
        """
        logger.info("Consolidation de la mémoire épisodique...")
        
        # Appliquer la décroissance d'importance pour les vieux épisodes
        now = datetime.now()
        for episode in self.episodes.values():
            age_days = (now - episode.start_time).days
            if age_days > 0:
                decay_factor = self.importance_decay ** age_days
                episode.importance *= decay_factor
        
        self.stats["consolidations"] += 1
        logger.info(f"Consolidation terminée (#{self.stats['consolidations']})")
    
    def _prune_episodes(self):
        """Élaguer les épisodes les moins importants"""
        # Trier par importance et date
        sorted_episodes = sorted(
            self.episodes.values(),
            key=lambda e: (e.importance, e.start_time),
            reverse=False
        )
        
        # Supprimer les moins importants
        to_remove = len(self.episodes) - self.max_episodes
        for episode in sorted_episodes[:to_remove]:
            del self.episodes[episode.episode_id]
            self.stats["episodes_pruned"] += 1
            logger.debug(f"Épisode élagué: {episode.episode_id} (importance={episode.importance:.2f})")
        
        # Reconstruire les index
        if to_remove > 0:
            self._rebuild_indexes()
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Récupérer un épisode spécifique"""
        return self.episodes.get(episode_id)
    
    def get_recent_episodes(
        self,
        n: int = 10,
        min_importance: float = 0.0,
        agent_id: Optional[str] = None
    ) -> List[Episode]:
        """
        Récupérer les n épisodes les plus récents
        
        Args:
            n: Nombre d'épisodes
            min_importance: Importance minimum
            agent_id: Filtrer par agent
            
        Returns:
            Liste d'épisodes
        """
        episodes = list(self.episodes.values())
        
        # Filtrer par importance
        if min_importance > 0:
            episodes = [ep for ep in episodes if ep.importance >= min_importance]
        
        # Filtrer par agent
        if agent_id:
            episodes = [ep for ep in episodes if agent_id in ep.get_involved_agents()]
        
        # Trier par date de fin (plus récent d'abord)
        episodes.sort(key=lambda e: e.end_time or datetime.now(), reverse=True)
        
        return episodes[:n]
    
    def search_by_tag(self, tag: str) -> List[Episode]:
        """Rechercher des épisodes par tag"""
        tag = tag.lower()
        episode_ids = self._episodes_by_tag.get(tag, [])
        return [self.episodes[eid] for eid in episode_ids if eid in self.episodes]
    
    def search_by_agent(self, agent_id: str, n: Optional[int] = None) -> List[Episode]:
        """Rechercher des épisodes impliquant un agent"""
        episode_ids = self._episodes_by_agent.get(agent_id, [])
        episodes = [self.episodes[eid] for eid in episode_ids if eid in self.episodes]
        
        # Trier par date récente
        episodes.sort(key=lambda e: e.end_time or datetime.now(), reverse=True)
        
        return episodes[:n] if n else episodes
    
    def search_by_outcome(self, outcome: str) -> List[Episode]:
        """Rechercher des épisodes par résultat"""
        episode_ids = self._episodes_by_outcome.get(outcome, [])
        return [self.episodes[eid] for eid in episode_ids if eid in self.episodes]
    
    def search_by_content(
        self,
        query: str,
        n: int = 10,
        search_in_events: bool = True
    ) -> List[Episode]:
        """
        Rechercher des épisodes par contenu
        
        Args:
            query: Requête de recherche
            n: Nombre maximum de résultats
            search_in_events: Chercher aussi dans les événements
            
        Returns:
            Liste d'épisodes correspondants
        """
        results = []
        query_lower = query.lower()
        
        for episode in self.episodes.values():
            score = 0
            
            # Chercher dans le résumé
            if episode.summary and query_lower in episode.summary.lower():
                score += 10
            
            # Chercher dans les tags
            for tag in episode.tags:
                if query_lower in tag:
                    score += 5
            
            # Chercher dans les métadonnées
            metadata_str = json.dumps(episode.metadata).lower()
            if query_lower in metadata_str:
                score += 3
            
            # Chercher dans les événements si demandé
            if search_in_events:
                for event in episode.events:
                    content_str = str(event.get("content", "")).lower()
                    if query_lower in content_str:
                        score += 1
            
            if score > 0:
                results.append({
                    "episode": episode,
                    "score": score
                })
        
        # Trier par score et importance
        results.sort(
            key=lambda x: (x["score"], x["episode"].importance),
            reverse=True
        )
        
        return [r["episode"] for r in results[:n]]
    
    def get_timeline(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[List[str]] = None
    ) -> List[Episode]:
        """
        Obtenir une timeline d'épisodes
        
        Args:
            start_time: Début de la période
            end_time: Fin de la période
            tags: Filtrer par tags
            
        Returns:
            Liste d'épisodes triés chronologiquement
        """
        episodes = list(self.episodes.values())
        
        # Filtrer par période
        if start_time:
            episodes = [e for e in episodes if (e.end_time or datetime.now()) >= start_time]
        
        if end_time:
            episodes = [e for e in episodes if e.start_time <= end_time]
        
        # Filtrer par tags
        if tags:
            tags_lower = [t.lower() for t in tags]
            episodes = [e for e in episodes if any(t in e.tags for t in tags_lower)]
        
        # Trier chronologiquement
        episodes.sort(key=lambda e: e.start_time)
        
        return episodes
    
    def get_context_summary(
        self,
        n_recent: int = 5,
        agent_id: Optional[str] = None,
        include_key_moments: bool = True
    ) -> str:
        """
        Obtenir un résumé de contexte des épisodes récents
        Formaté pour être utilisé comme contexte dans un prompt LLM
        
        Args:
            n_recent: Nombre d'épisodes récents
            agent_id: Filtrer par agent
            include_key_moments: Inclure les moments clés
            
        Returns:
            Résumé formaté
        """
        episodes = self.get_recent_episodes(n=n_recent, agent_id=agent_id)
        
        if not episodes:
            return "Aucun épisode récent disponible."
        
        context = "=== Contexte Épisodique Récent ===\n\n"
        
        for i, episode in enumerate(episodes, 1):
            context += f"[Épisode {i}] {episode.episode_id}\n"
            context += f"  Période: {episode.start_time.strftime('%Y-%m-%d %H:%M')}"
            
            if episode.end_time:
                context += f" - {episode.end_time.strftime('%H:%M')}"
            context += f" ({episode.duration_seconds():.0f}s)\n"
            
            if episode.summary:
                context += f"  Résumé: {episode.summary}\n"
            
            if episode.tags:
                context += f"  Tags: {', '.join(list(episode.tags)[:5])}\n"
            
            context += f"  Événements: {len(episode.events)}, "
            context += f"Importance: {episode.importance:.2f}"
            
            if episode.outcome:
                context += f", Résultat: {episode.outcome}"
            
            context += "\n"
            
            # Moments clés
            if include_key_moments and len(episode.events) > 0:
                key_moments = episode.get_key_moments(n=2)
                if key_moments:
                    context += "  Moments clés:\n"
                    for moment in key_moments:
                        context += f"    - [{moment['type']}] {str(moment['content'])[:60]}...\n"
            
            context += "\n"