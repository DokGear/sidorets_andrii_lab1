"""Main application entry point for the Audio Recorder CLI tool."""

import logging
import sys
from enum import IntEnum

from audio_engine import AudioEngine, AudioEngineError
from cli import parse_args
from logger import setup_logging
from plotter import Plotter, PlotterError
from process import AudioProcessor, AudioProcessingError

class ExitCode(IntEnum):
    SUCCESS = 0
    FAILURE = 1
    INVALID_CONFIG = 2
    INTERRUPTED = 130


def main() -> int:
    setup_logging()
    logger = logging.getLogger("main")

    try:
        app_config = parse_args()
        engine = AudioEngine(app_config.audio)
        plotter = Plotter(app_config.plot)
        processor = AudioProcessor(app_config.audio)

        logger.info("=== ЕТАП 1: Запис звуку ===")
        audio_data = engine.record()
        logger.info("Запис завершено. Отримано семплів: %d", len(audio_data))

        logger.info("=== ЕТАП 2: Збереження ОРИГІНАЛУ та графіків ===")
        saved_original = engine.save(audio_data)
        
        plot_original = plotter.plot_signal(
            audio_data=audio_data,
            sample_rate=app_config.audio.sample_rate,
            audio_filename=app_config.audio.output_filename,
        )
        spec_original = plotter.plot_spectrogram(
            audio_data=audio_data,
            sample_rate=app_config.audio.sample_rate,
            audio_filename=app_config.audio.output_filename,
        )
        logger.info("Оригінал збережено: %s, Спектрограма: %s", plot_original, spec_original)

        logger.info("=== ЕТАП 3: Обробка (обрізання) ===")
        processed_data = processor.trim_silence(audio_data)
        logger.info("Обробка завершена. Семплів після обробки: %d", len(processed_data))

        logger.info("=== ЕТАП 4: Збереження ОБРОБЛЕНОГО та графіків ===")
        processed_name = f"{app_config.audio.output_filename}_processed"
        saved_processed = engine.save(processed_data, filename_suffix="_processed")
        
        plot_processed = plotter.plot_signal(
            audio_data=processed_data,
            sample_rate=app_config.audio.sample_rate,
            audio_filename=processed_name,
        )
        spec_processed = plotter.plot_spectrogram(
            audio_data=processed_data,
            sample_rate=app_config.audio.sample_rate,
            audio_filename=processed_name,
        )
        logger.info("Оброблений файл збережено: %s, Спектрограма: %s", plot_processed, spec_processed)

        return ExitCode.SUCCESS

    except KeyboardInterrupt:
        logger.warning("\nRecording process aborted by user.")
        return ExitCode.INTERRUPTED
    except ValueError as err:
        logger.error("Configuration error: %s", err)
        return ExitCode.INVALID_CONFIG
    except (AudioEngineError, PlotterError, AudioProcessingError) as err:
        logger.error("Application error: %s", err)
        return ExitCode.FAILURE
    except Exception as err:
        logger.critical("Unexpected failure: %s", err, exc_info=True)
        return ExitCode.FAILURE


if __name__ == "__main__":
    sys.exit(main())