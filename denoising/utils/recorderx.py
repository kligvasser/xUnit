import utils.misc as misc
from tensorboardX import SummaryWriter

class RecoderX():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(logdir=log_dir)
        self.log = ''

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step, walltime=walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.writer.add_scalars(main_tag=main_tag, tag_scalar_dict=tag_scalar_dict, global_step=global_step, walltime=walltime)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        self.writer.add_image(tag=tag, img_tensor=img_tensor, global_step=global_step, walltime=walltime, dataformats=dataformats)

    def add_image_grid(self, tag, img_tensor, nrow=8, global_step=None, padding=0, pad_value=0,walltime=None, dataformats='CHW'):
        grid = misc.make_image_grid(img_tensor, nrow, padding=padding, pad_value=pad_value)
        self.writer.add_image(tag=tag, img_tensor=grid, global_step=global_step, walltime=walltime, dataformats=dataformats)

    def add_graph(self, graph_profile, walltime=None):
        self.writer.add_graph(graph_profile, walltime=walltime)

    def add_histogram(self, tag, values, global_step=None):
        self.writer.add_histogram(tag, values, global_step)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        self.writer.add_figure(tag, figure, global_step=global_step, close=close, walltime=walltime)

    def export_json(self, out_file):
        self.writer.export_scalars_to_json(out_file)

    def close(self):
        self.writer.close()

if __name__ == "__main__":
    print('None')
