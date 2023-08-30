class Model:
    def __init__(self):
        pass

    def __call__(self, input_ids: torch.Tensor) -> Distribution:
        raise NotImplementedError()
    
    def decode(self, input_ids: torch.Tensor) -> str:
        raise NotImplementedError()