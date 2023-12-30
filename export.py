import torch
from reco_encoder.model import model


if __name__ == "__main__":
    encoder = model.AutoEncoder(layer_sizes=[19547, 256, 128], is_constrained=True)
    x = torch.randn(64, 19547)
    try:
        torch.export.export(encoder, (x,))
        print("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print("[JIT] torch.export failed.")
        raise e
