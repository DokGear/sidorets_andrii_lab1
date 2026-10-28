"""Visualization module for rendering audio signal waveforms and spectrograms."""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from config import PlotConfig

logger = logging.getLogger(__name__)


class PlotterError(Exception):
    """Raised when audio signal visualization fails."""


class Plotter:
    """Renders time-domain audio signal plots and spectrograms using Matplotlib."""

    def __init__(self, config: PlotConfig) -> None:
        self.config = config

    def plot_signal(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        audio_filename: str = "output.wav",
    ) -> Path | None:
        """Plot and save the time-domain audio waveform."""
        if not self.config.enabled:
            logger.info("Plotting is disabled by configuration.")
            return None

        if audio_data is None or audio_data.size == 0:
            raise PlotterError("Cannot plot empty audio data.")

        logger.info("Generating audio signal (waveform) plot...")

        try:
            num_samples = len(audio_data)
            time_axis = np.linspace(0, num_samples / sample_rate, num_samples)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_axis, audio_data, color="#1f77b4", alpha=0.85, linewidth=1.0)
            ax.set_title(f"Audio Signal: {Path(audio_filename).name}", fontsize=12)
            ax.set_xlabel("Time (seconds)", fontsize=10)
            ax.set_ylabel("Amplitude", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.tight_layout()

            target_path = self.config.get_figure_path(audio_filename)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_path, dpi=self.config.dpi)
            
            resolved_path = target_path.resolve()
            logger.info("Waveform image successfully saved to: %s", resolved_path)

            if self.config.show:
                plt.show()

            plt.close(fig)
            return resolved_path

        except Exception as err:
            logger.error("Failed to render audio plot: %s", err)
            raise PlotterError(f"Could not render audio signal plot: {err}") from err

    def plot_spectrogram(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        audio_filename: str = "output.wav",
    ) -> Path | None:
        """Generate and save a spectrogram of the audio signal."""
        if not self.config.enabled:
            return None

        if audio_data is None or audio_data.size == 0:
            raise PlotterError("Cannot plot empty audio data.")

        logger.info("Generating audio spectrogram...")

        try:
            if audio_data.ndim > 1:
                data = audio_data[:, 0]
            else:
                data = audio_data

            fig, ax = plt.subplots(figsize=(10, 4))
            
            pxx, freqs, bins, im = ax.specgram(data, NFFT=1024, Fs=sample_rate,
                                               noverlap=512, cmap="viridis")
            
            ax.set_title(f"Spectrogram: {Path(audio_filename).name}", fontsize=12)
            ax.set_xlabel("Time (seconds)", fontsize=10)
            ax.set_ylabel("Frequency (Hz)", fontsize=10)
            
            fig.colorbar(im, ax=ax, label="Intensity (dB)")
            fig.tight_layout()

            target_path = self.config.get_figure_path(audio_filename)
            target_path = target_path.with_name(f"{target_path.stem}_spectrogram{target_path.suffix}")
            
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_path, dpi=self.config.dpi)
            
            resolved_path = target_path.resolve()
            logger.info("Spectrogram image successfully saved to: %s", resolved_path)

            if self.config.show:
                plt.show()

            plt.close(fig)
            return resolved_path

        except Exception as err:
            logger.error("Failed to render spectrogram: %s", err)
            raise PlotterError(f"Could not render spectrogram: {err}") from err