import torch
from torch.utils.data import Dataset

class DummyMultiModalDataset(Dataset):
    def __init__(self, num_samples=1000, label_dim=14, image_dim=1024):
        super().__init__()
        self.num_samples = num_samples
        self.label_dim = label_dim
        self.image_dim = image_dim

        # just random captions
        self.caption_templates = [
            "A sample caption",
            "Another caption",
            "Random description",
            "Synthetic example",
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = {
            "labels": torch.randn(self.label_dim),     # [14]
            "image_embeds": torch.randn(self.image_dim),  # [1024]
            "captions": self.caption_templates[idx % len(self.caption_templates)],
        }
        return sample
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = DummyMultiModalDataset(num_samples=2)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(dataloader))
    print(batch)
    # Example output:
    # {
    #   'labels': torch.Size([2, 14]),
    #   'image_embeds': torch.Size([2, 1024]),
    #   'captions': list of 2 strings
    # }