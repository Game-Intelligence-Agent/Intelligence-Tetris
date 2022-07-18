import os
from torch.utils.tensorboard import SummaryWriter

class tb_handler:

    def __init__(self, path: str, name: str, model) -> None:

        if path[-1] != '/':
            path += '/'

        if not os.path.exists(path):
            os.mkdir(path)
        
        self.writer = SummaryWriter(f'{path}{name}')
        self.model = model

    def add_image(self, image, name):

        self.writer.add_image(name, image)
        self.writer.flush()

    def add_graph(self, input_data):
        self.writer.add_graph(self.model, input_data)
        self.writer.flush()

    def add_scalar(self, data, step, tag):
        self.writer.add_scalar(tag = tag, scalar_value = data, global_step = step)
        self.writer.flush()

    def show_params(self, global_step, mode = ['weight', 'grad']):

        for name, param in self.model.named_parameters():
            # print(f'{name}: {param.grad}')
            if 'weight' in mode:
                self.add_histogram(param, global_step, name + '_weight')
            if 'grad' in mode:
                self.add_histogram(param.grad, global_step, name + '_grad')
        self.writer.flush()


    def add_histogram(self, weight, global_step, name):
        self.writer.add_histogram(tag = name, values = weight, global_step = global_step)
        self.writer.flush()