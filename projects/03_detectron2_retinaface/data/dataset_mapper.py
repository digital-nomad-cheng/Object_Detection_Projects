from detectron2.data.dataset_mapper import DatasetMapper as BaseDatasetMapper

from . import detection_utils

class DatasetMapper(BaseDatasetMapper):

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        self.tfm_gens = dettection_utils.build_transform_gen(cfg, is_train)
