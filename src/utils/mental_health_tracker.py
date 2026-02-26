import time
from collections import deque

class MentalHealthTracker:
    """
    Tracks emotional fluctuations over a session to provide
    a comprehensive psychological summary.
    Useful for therapy settings or sustained user interaction.
    """
    def __init__(self, history_limit=100):
        # Store tuples of (timestamp, emotion, confidence_dict)
        self.history = deque(maxlen=history_limit)
        self.session_start = time.time()
        
        self.emotion_weights = {
            'Angry': -1.0,
            'Disgust': -0.8,
            'Fear': -0.8,
            'Sad': -0.7,
            'Neutral': 0.0,
            'Surprise': 0.3,
            'Happy': 1.0
        }

    def add_reading(self, primary_emotion, confidence_scores):
        """Adds a new reading to the temporal history."""
        self.history.append((time.time(), primary_emotion, confidence_scores))

    def generate_summary(self):
        """
        Analyzes the temporal data to generate a user-friendly
        psychological state summary.
        """
        if not self.history:
            return {"status": "insufficient_data"}
            
        time_elapsed = time.time() - self.session_start
        
        # Calculate dominant emotion over time
        emotion_counts = {}
        total_valence = 0.0
        
        for _, emotion, scores in self.history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            # Calculate emotional valence (positivity/negativity score (-1 to 1))
            for emo, conf in scores.items():
                total_valence += (self.emotion_weights.get(emo, 0) * conf)

        avg_valence = total_valence / len(self.history)
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        stability_score = emotion_counts[dominant_emotion] / len(self.history)
        
        # Generate Clinical Insight String
        if avg_valence < -0.3:
            insight = "User exhibits sustained negative affect. Potential stress or depressive markers present."
            state = "Distressed"
        elif avg_valence > 0.3:
            insight = "User exhibits sustained positive affect. Generally healthy emotional state."
            state = "Uplifted"
        else:
            if stability_score < 0.4:
                insight = "High emotional volatility detected. User state is rapidly fluctuating."
                state = "Volatile"
            else:
                insight = "User exhibits baseline emotional stability."
                state = "Stable (Neutral)"

        return {
            "session_duration_seconds": round(time_elapsed, 1),
            "total_readings": len(self.history),
            "dominant_emotion": dominant_emotion,
            "emotional_stability": round(stability_score, 2),
            "average_valence": round(avg_valence, 2),
            "clinical_state": state,
            "clinical_insight": insight
        }

# Global instance for demonstration purposes (in prod, use DB/Sessions per user)
global_tracker = MentalHealthTracker()

if __name__ == "__main__":
    print("Mental Health Temporal Tracker initialized.")
