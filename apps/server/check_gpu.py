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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("tiny", device=device, compute_type=compute_type)
        print(
            f"Initialized faster-whisper model on {device} ({compute_type}) successfully."
        )
        try:
            # Some versions expose a device attribute; if not, this will be skipped.
            device_attr = getattr(model, "device", f"{device} (assumed)")
            print("faster-whisper device:", device_attr)
        except Exception:
            print(
                f"faster-whisper initialized on {device} ({compute_type}) – device attribute not available."
            )
    except Exception as e:
        print(f"Failed to initialize faster-whisper on {device}:", repr(e))
    print()


def check_kokoro():
    print("=== Kokoro TTS ===")
    if KPipeline is None:
        print("Kokoro TTS not installed.")
        print()
        return

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = KPipeline(lang_code="a", device=device)
        print(f"Initialized Kokoro KPipeline with device='{device}' successfully.")
    except Exception as e:
        print(f"Failed to initialize Kokoro KPipeline on {device}:", repr(e))
    print()


def main():
    check_torch()
    check_faster_whisper()
    check_kokoro()


if __name__ == "__main__":
    main()
