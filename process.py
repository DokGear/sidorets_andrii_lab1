import logging
import numpy as np
from config import AudioConfig

logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    pass

class AudioProcessor:
    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    def trim_silence(
        self,
        audio_data: np.ndarray,
        threshold_db: float = -40.0,
        padding_ms: float = 50.0,
    ) -> np.ndarray:
        if audio_data is None or audio_data.size == 0:
            raise AudioProcessingError("Cannot process empty audio data.")

        logger.info("Trimming silence (threshold=%.1f dB)...", threshold_db)

        try:
            envelope = np.abs(audio_data)
            if envelope.ndim > 1:
                envelope = envelope.max(axis=1)

            threshold = 10 ** (threshold_db / 20.0)
            mask = envelope > threshold
            
            if not mask.any():
                logger.warning("No samples above threshold; returning an unmodified copy.")
                return audio_data.copy()

            first = int(np.argmax(mask))
            last = int(len(mask) - np.argmax(mask[::-1]))

            pad = int(self.config.sample_rate * padding_ms / 1000.0)
            start = max(0, first - pad)
            end = min(len(audio_data), last + pad)

            trimmed = audio_data[start:end].copy()
            
            if len(trimmed) < (self.config.sample_rate * 0.1):
                logger.warning("Trimmed audio is too short. Returning original data.")
                return audio_data.copy()

            logger.info(
                "Trimmed %d -> %d samples (%.2fs -> %.2fs).",
                len(audio_data),
                len(trimmed),
                len(audio_data) / self.config.sample_rate,
                len(trimmed) / self.config.sample_rate,
            )
            return trimmed
            
        except Exception as err:
            logger.error("Unexpected error during signal processing: %s", err)
            raise AudioProcessingError(f"Failed to process audio: {err}") from err