from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner

class RealPushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir):
        super().__init__(output_dir)
    
    def run(self, policy: BaseImagePolicy):
        return dict()