import torch

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from kokoro import KPipeline
except ImportError:
    KPipeline = None


def check_torch():
    print("=== PyTorch ===")
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        print("Current CUDA device index:", device_index)
        print("Current CUDA device name:", torch.cuda.get_device_name(device_index))
    print()


def check_faster_whisper():
    print("=== faster-whisper ===")
    if WhisperModel is None:
        print("faster-whisper not installed.")
        print()
        return

    try:
        # Use a small model to keep the check lightweight
        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        print("Initialized faster-whisper model successfully.")
        try:
            # Some versions expose a device attribute; if not, this will be skipped.
            device = getattr(model, "device", "cuda (assumed)")
            print("faster-whisper device:", device)
        except Exception:
            print(
                "faster-whisper initialized on CUDA (float16) – device attribute not available."
            )
    except Exception as e:
        print("Failed to initialize faster-whisper on CUDA:", repr(e))
    print()


def check_kokoro():
    print("=== Kokoro TTS ===")
    if KPipeline is None:
        print("Kokoro TTS not installed.")
        print()
        return

    try:
        pipeline = KPipeline(lang_code="a", device="cuda")
        print("Initialized Kokoro KPipeline with device='cuda' successfully.")
    except Exception as e:
        print("Failed to initialize Kokoro KPipeline on CUDA:", repr(e))
    print()


def main():
    check_torch()
    check_faster_whisper()
    check_kokoro()


if __name__ == "__main__":
    main()
