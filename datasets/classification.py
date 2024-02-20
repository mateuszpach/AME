import abc
import os
import sys
from collections import Counter
from typing import Optional
import json

import torch
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchvision.datasets import ImageNet

from datasets.base import BaseDataModule
from datasets.utils import get_default_img_transform, get_default_aug_img_transform


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.data = file_list
        self.labels = label_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.open(sample).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]

    def class_stats(self):
        return [v for k, v in sorted(Counter(self.labels).items())]


class BaseClassificationDataModule(BaseDataModule, abc.ABC):
    cls_num_classes = 0

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.inst_num_classes = None

    @property
    def num_classes(self):
        if self.inst_num_classes is not None:
            return self.inst_num_classes
        return self.cls_num_classes

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        print(f'Train class statistics:', self.train_dataset.class_stats(), file=sys.stderr)
        return super().train_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        print(f'Test class statistics:', self.test_dataset.class_stats(), file=sys.stderr)
        return super().test_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        print(f'Val class statistics:', self.val_dataset.class_stats(), file=sys.stderr)
        return super().val_dataloader()


class Sun360Classification(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 26

    def setup(self, stage: Optional[str] = None) -> None:
        with open(os.path.join(os.path.dirname(__file__), 'meta/sun360-dataset26-random.txt')) as f:
            file_list = f.readlines()
        labels = [str.join('/', p.split('/')[:2]) for p in file_list]
        classes = {name: idx for idx, name in enumerate(sorted(set(labels)))}
        labels = [classes[x] for x in labels]
        file_list = [os.path.join(self.data_dir, p.strip()) for p in file_list]
        val_list = file_list[:len(file_list) // 10]
        val_labels = labels[:len(file_list) // 10]
        train_list = file_list[len(file_list) // 10:]
        train_labels = labels[len(file_list) // 10:]

        if stage == 'fit':
            self.train_dataset = ClassificationDataset(file_list=train_list, label_list=train_labels,
                                                       transform=
                                                       get_default_img_transform(self.image_size)
                                                       if self.no_aug else
                                                       get_default_aug_img_transform(self.image_size, scale=False))
            self.val_dataset = ClassificationDataset(file_list=val_list, label_list=val_labels,
                                                     transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()


class ImageNetWithStats(ImageNet):
    def class_stats(self):
        return [v for k, v in sorted(Counter(self.targets).items())]


class ImageNet1kClassification(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 1000

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == 'fit':
            self.train_dataset = ImageNetWithStats(root=self.data_dir, split='train',
                                                   transform=
                                                   get_default_img_transform(self.image_size)
                                                   if self.no_aug else
                                                   get_default_aug_img_transform(self.image_size, scale=False))
            self.val_dataset = ImageNetWithStats(root=self.data_dir, split='val',
                                                 transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()


class FMoVClassification(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 62

    def _fix_path(self, path: str) -> str:
        split = os.path.normpath(path).split(os.sep)
        stripped = os.sep.join(split[split.index('input'):])
        return os.path.join(self.data_dir, stripped)

    def setup(self, stage: Optional[str] = None) -> None:
        with open(os.path.join(self.data_dir, 'working/training_struct.json')) as f:
            train_samples = json.load(f)

        train_files = [self._fix_path(sample['img_path']) for sample in train_samples]
        train_labels = [sample['category'] - 1 for sample in train_samples]

        with open(os.path.join(self.data_dir, 'working/test_struct.json')) as f:
            test_samples = json.load(f)
        test_files = [self._fix_path(sample['img_path']) for sample in test_samples]

        with open(os.path.join(self.data_dir, 'working/test_gt_mapping.json')) as f:
            test_gt_mapping = json.load(f)

        caption_to_label = {
            "airport": 0, "airport_hangar": 1, "airport_terminal": 2, "amusement_park": 3, "aquaculture": 4,
            "archaeological_site": 5, "barn": 6, "border_checkpoint": 7, "burial_site": 8, "car_dealership": 9,
            "construction_site": 10, "crop_field": 11, "dam": 12, "debris_or_rubble": 13, "educational_institution": 14,
            "electric_substation": 15, "factory_or_powerplant": 16, "fire_station": 17, "flooded_road": 18,
            "fountain": 19, "gas_station": 20, "golf_course": 21, "ground_transportation_station": 22, "helipad": 23,
            "hospital": 24, "impoverished_settlement": 25, "interchange": 26, "lake_or_pond": 27, "lighthouse": 28,
            "military_facility": 29, "multi-unit_residential": 30, "nuclear_powerplant": 31, "office_building": 32,
            "oil_or_gas_facility": 33, "park": 34, "parking_lot_or_garage": 35, "place_of_worship": 36,
            "police_station": 37, "port": 38, "prison": 39, "race_track": 40, "railway_bridge": 41,
            "recreational_facility": 42, "road_bridge": 43, "runway": 44, "shipyard": 45, "shopping_mall": 46,
            "single-unit_residential": 47, "smokestack": 48, "solar_farm": 49, "space_facility": 50, "stadium": 51,
            "storage_tank": 52, "surface_mine": 53, "swimming_pool": 54, "toll_booth": 55, "tower": 56,
            "tunnel_opening": 57, "waste_disposal": 58, "water_treatment_facility": 59, "wind_farm": 60, "zoo": 61}

        id_to_caption = {}
        for mapping in test_gt_mapping:
            for box_mapping in mapping['box_mapping']:
                id_to_caption[mapping['output'][5:] + '/' + str(box_mapping['output'])] = box_mapping['label']

        test_labels = []
        for sample in test_samples:
            id = os.sep.join(os.path.normpath(sample['img_path']).split(os.sep)[-3:-1])
            caption = id_to_caption[id]
            if caption != 'false_detection':
                label = caption_to_label[caption]
                test_labels.append(label)

        if stage == 'fit':
            self.train_dataset = ClassificationDataset(file_list=train_files, label_list=train_labels,
                                                       transform=
                                                       get_default_img_transform(self.image_size)
                                                       if self.no_aug else
                                                       get_default_aug_img_transform(self.image_size, scale=False))
            self.val_dataset = ClassificationDataset(file_list=test_files, label_list=test_labels,
                                                     transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()


class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        data = torch.load(file_name)
        self.data = data['latents']
        self.labels = data['targets']

        print(self.class_stats())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].unsqueeze(0), self.labels[index]

    def class_stats(self):
        return [v for k, v in sorted(Counter(x.item() for x in self.labels).items())]


class EmbedClassification(BaseClassificationDataModule):
    has_test_data = False

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.inst_num_classes = args.num_classes

    @classmethod
    def add_argparse_args(cls, parent_parser, **kwargs):
        parent_parser = super().add_argparse_args(parent_parser, **kwargs)
        parser = parent_parser.add_argument_group(EmbedClassification.__name__)
        parser.add_argument('--num-classes',
                            help='number of classes',
                            type=int,
                            default=26)
        return parent_parser

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = EmbedDataset(file_name=os.path.join(self.data_dir, 'embeds_train.pck'))
            self.val_dataset = EmbedDataset(file_name=os.path.join(self.data_dir, 'embeds_val.pck'))
        else:
            raise NotImplemented()
