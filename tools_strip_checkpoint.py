from ultralytics.utils.torch_utils import strip_optimizer


if __name__ == "__main__":
    strip_optimizer("models/best.pt", "models/best-finetune.pt")
