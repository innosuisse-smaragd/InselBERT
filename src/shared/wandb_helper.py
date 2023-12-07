import wandb


class WandbHelper:
    def __init__(self, project_name, config):
        self.config = config
        self.project_name = project_name
        wandb.init(project=project_name, config=config)

    @staticmethod
    def finish():
        wandb.finish()

    @staticmethod
    def log(**kwargs):
        wandb.log(kwargs)
