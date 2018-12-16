from tensorboardX import SummaryWriter

class LogSummary(object):

    def __init__(self, log_path):

        self.writer = SummaryWriter(log_path)

    def write_scalars(self, scalars, names, n_iter, tag=None):

        for scalar, name in zip(scalars, names):
            if tag is not None:
                name = '/'.join([tag, name])
            self.writer.add_scalar(name, scalar, n_iter)

    def write_hist_parameters(self, net, n_iter):
        for name, param in net.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)
