class VisualisationNotYetSupportedError(Exception):
    def __init__(self, method_name, model_name):
        super().__init__(f"{method_name} is not implemented yet for {model_name}")
