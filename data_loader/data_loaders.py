from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from data_loader.CE_dataset import CE
from utils.util import HashableOrderedDict


def dataset_loader(dataset_name, data_dir, categories, expert_dims, split, text_dim, text_feat,
                   max_text_words, max_expert_tokens, use_val=False):
    dataset_classes = {
        "CE": CE
    }
    if len(categories) >= 1 and split == 'train':
        dataset_list = []
        for cat in categories:
            dataset = dataset_classes[dataset_name](
                data_dir=data_dir,
                text_dim=text_dim,
                category=cat,
                expert_dims=expert_dims,
                split=split,
                text_feat=text_feat,
                max_text_words=max_text_words,
                max_expert_tokens=max_expert_tokens,
                transforms=transforms.Compose([
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
            )
            dataset_list.append(dataset)

        if use_val: # use valset as training
            for cat in categories:
                dataset = dataset_classes[dataset_name](
                    data_dir=data_dir,
                    text_dim=text_dim,
                    category=cat,
                    expert_dims=expert_dims,
                    split='val',
                    text_feat=text_feat,
                    max_text_words=max_text_words,
                    max_expert_tokens=max_expert_tokens,
                    transforms=transforms.Compose([
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])
                )
                dataset_list.append(dataset)

        dataset = ConcatDataset(dataset_list)
    elif split in ['val', 'val_trg', 'test', 'test_trg']:
        dataset_list = []
        for cat in categories:
            dataset = dataset_classes[dataset_name](
                data_dir=data_dir,
                text_dim=text_dim,
                category=cat,
                expert_dims=expert_dims,
                split=split,
                text_feat=text_feat,
                max_text_words=max_text_words,
                max_expert_tokens=max_expert_tokens,
                transforms=transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
            )
            dataset_list.append(dataset)
        dataset = dataset_list
    
    return dataset


class ExpertDataLoader:

    def __init__(self, dataset_name, data_dir, categories, expert_dims, num_workers,
                 batch_size, text_feat, text_dim, max_text_words, max_expert_tokens, mode,
                 use_val=False):

        expert_dims = HashableOrderedDict(expert_dims)
        self.dataloaders = {}

        if mode == 'train':
            train_dataset = dataset_loader(
                dataset_name=dataset_name,
                data_dir=data_dir,
                categories=categories,
                expert_dims=expert_dims,
                split='train',
                text_dim=text_dim,
                text_feat=text_feat,
                max_text_words=max_text_words,
                max_expert_tokens=max_expert_tokens,
                use_val=use_val
            )

            val_datasets = dataset_loader(
                dataset_name=dataset_name,
                data_dir=data_dir,
                categories=categories,
                expert_dims=expert_dims,
                split='val',
                text_dim=text_dim,
                text_feat=text_feat,
                max_text_words=max_text_words,
                max_expert_tokens=max_expert_tokens,
            )

            val_trg_datasets = dataset_loader(
                dataset_name=dataset_name,
                data_dir=data_dir,
                categories=categories,
                expert_dims=expert_dims,
                split='val_trg',
                text_dim=text_dim,
                text_feat=text_feat,
                max_text_words=max_text_words,
                max_expert_tokens=max_expert_tokens,
            )

            if isinstance(train_dataset, ConcatDataset):
                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=train_dataset.datasets[0].collate_fn,
                    drop_last=True,
                    shuffle=True,
                )

            else:
                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=train_dataset.collate_fn,
                    drop_last=True,
                    shuffle=True,
                )

            self.dataloaders['train'] = train_loader

            for dataset in val_datasets:
                self.dataloaders[dataset.category] = DataLoader(
                    dataset=dataset,
                    batch_size=4,
                    num_workers=num_workers,
                    collate_fn=dataset.collate_fn,
                    drop_last=False,
                    shuffle=False
                )

            for dataset in val_trg_datasets:
                self.dataloaders[dataset.category + '_trg'] = DataLoader(
                    dataset=dataset,
                    batch_size=100,
                    num_workers=num_workers,
                    collate_fn=dataset.collate_fn,
                    drop_last=False,
                    shuffle=False
                )
        elif mode == 'val':
            test_datasets = dataset_loader(
                dataset_name=dataset_name,
                data_dir=data_dir,
                categories=categories,
                expert_dims=expert_dims,
                split='val',
                text_dim=text_dim,
                text_feat=text_feat,
                max_text_words=max_text_words,
                max_expert_tokens=max_expert_tokens,
            )

            test_trg_datasets = dataset_loader(
                dataset_name=dataset_name,
                data_dir=data_dir,
                categories=categories,
                expert_dims=expert_dims,
                split='val_trg',
                text_dim=text_dim,
                text_feat=text_feat,
                max_text_words=max_text_words,
                max_expert_tokens=max_expert_tokens,
            )

            for dataset in test_datasets:
                self.dataloaders[dataset.category] = DataLoader(
                    dataset=dataset,
                    batch_size=8,
                    num_workers=num_workers,
                    collate_fn=dataset.collate_fn,
                    drop_last=False,
                    shuffle=False
                )

            for dataset in test_trg_datasets:
                self.dataloaders[dataset.category + '_trg'] = DataLoader(
                    dataset=dataset,
                    batch_size=100,
                    num_workers=num_workers,
                    collate_fn=dataset.collate_fn,
                    drop_last=False,
                    shuffle=False
                )

        elif mode == 'test':
            test_datasets = dataset_loader(
                dataset_name=dataset_name,
                data_dir=data_dir,
                categories=categories,
                expert_dims=expert_dims,
                split='test',
                text_dim=text_dim,
                text_feat=text_feat,
                max_text_words=max_text_words,
                max_expert_tokens=max_expert_tokens,
            )

            test_trg_datasets = dataset_loader(
                dataset_name=dataset_name,
                data_dir=data_dir,
                categories=categories,
                expert_dims=expert_dims,
                split='test_trg',
                text_dim=text_dim,
                text_feat=text_feat,
                max_text_words=max_text_words,
                max_expert_tokens=max_expert_tokens,
            )

            for dataset in test_datasets:
                self.dataloaders[dataset.category] = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=dataset.collate_fn,
                    drop_last=False,
                    shuffle=False
                )

            for dataset in test_trg_datasets:
                self.dataloaders[dataset.category + '_trg'] = DataLoader(
                    dataset=dataset,
                    batch_size=100,
                    num_workers=num_workers,
                    collate_fn=dataset.collate_fn,
                    drop_last=False,
                    shuffle=False
                )

        else:
            raise ValueError

        self.dataset_name = dataset_name

    def __getitem__(self, key):
        return self.dataloaders[key]
