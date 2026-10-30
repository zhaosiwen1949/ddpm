from datasets import load_dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Lambda
from torch.utils.data import DataLoader

dataset = load_dataset(
    "parquet",
    data_files={
        "test": "./datasets/fashion_mnist/test-00000-of-00001.parquet",
        "train": "./datasets/fashion_mnist/train-00000-of-00001.parquet"
    }
)
image_size =28
channels = 1
batch_size = 128

transform = Compose([
    RandomHorizontalFlip(),
    ToTensor(),
    Lambda(lambda t: (t * 2) - 1)
])

def data_tramsforms(examples):
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]
    return examples

transformed_dataset = dataset.with_transform(data_tramsforms).remove_columns("label")

dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

# batch = next(iter(dataloader))

# print(batch.keys())
