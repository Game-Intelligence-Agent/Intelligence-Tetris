from Models.Wrappers import Wrapper


class GNNWrapper(Wrapper):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_name: str,
        model_type: str,
        parameters_path: str = './Models/Parameters/',
        bias: bool = True,
        add_self_loop: bool = True,
        **kwargs
        ):

        Wrapper.__init__(self, in_channels,
            out_channels,
            model_name,
            model_type,
            parameters_path)