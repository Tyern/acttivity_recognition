from torch.utils.data import Dataset
from itertools import combinations

from . import *

# Update after 2023/11/11 houkoku
class CustomTrainDataset(Dataset):
    TRAIN_MODE = "train"
    TEST_MODE = "test"
    
    def __init__(self, mode, feature_data, label_data, missing_sensor_numbers=0):
        self.mode = mode
        assert mode in [self.TRAIN_MODE, self.TEST_MODE]
        
        self.features = feature_data
        self.label = label_data
        assert len(self.features) == len(self.label), "features len is not equal to label len"
        self.missing_sensor_numbers = missing_sensor_numbers

        self.missing_index_list = []
        for missing_count in range(missing_sensor_numbers + 1):
            for missing_index in combinations(range(SENSOR_NUM), missing_count):
                self.missing_index_list.append(missing_index)

    def transform(self, one_feature, missing_sensor_id_list):
        # Make one sensor data become 0
        one_feature_cp = one_feature.copy()
        
        for missing_sensor_id in missing_sensor_id_list:
            one_feature_cp[:, missing_sensor_id*6:(missing_sensor_id+1)*6] = 0
        return one_feature_cp
        
    def __len__(self):

        # take all available missing pattern * data number
        return len(self.features) * len(self.missing_index_list)
    
    def __getitem__(self, idx):
        # take all available missing pattern
        missing_sensor_id_list = self.missing_index_list[ idx // len(self.features) ]
        x = self.transform(self.features[ idx % len(self.features) ], missing_sensor_id_list)
        label = self.label[idx % len(self.features)]
        return x, int(label)
