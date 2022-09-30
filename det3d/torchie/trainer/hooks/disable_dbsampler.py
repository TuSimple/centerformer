from .hook import Hook


class DisableDBSamplerHook(Hook):
    def __init__(self, disable_dbsampler_after_epoch):
        self.disable_dbsampler_after_epoch = disable_dbsampler_after_epoch

    def before_epoch(self, trainer):
        if trainer.epoch >= self.disable_dbsampler_after_epoch:
            for pipeline in trainer.data_loader.dataset.pipeline.transforms:
                if "db_sampler" in dir(pipeline):
                    pipeline.db_sampler = None
