import argparse
import numpy as np
import os
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from constants import *
from dataset_factory import get_num_classes, get_dataset
from model_factory import get_architecture

parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("arch", type=str, choices=ARCHITECTURES, help="model name")
parser.add_argument("root", type=str, help="dir with checkpoint")
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument("--batch", type=int, default=100, help="batch size")
parser.add_argument("--max", type=int, default=-1, help="stop after this many batches")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
SEED = 742

# init seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)


def main():
    # load the classifier
    checkpoint = torch.load(os.path.join(args.root, 'checkpoint.pth.tar'))
    classifier = get_architecture(checkpoint["arch"], args.dataset, load_weights=False, use_cuda=USE_CUDA)
    classifier.load_state_dict(checkpoint['state_dict'])
    classifier.eval()

    # prepare dataset
    num_classes = get_num_classes(args.dataset)
    pin_memory = (args.dataset == "imagenet")
    input_size = 600 if checkpoint['arch'] == EFFICENTNETB7 else 224
    print(f'input size = {input_size}')
    dataset = get_dataset(args.dataset, args.split, target_class_counts=None, input_size=input_size)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch, num_workers=args.workers,
                            pin_memory=pin_memory)

    # dataframe for predictions
    logits = np.empty(shape=(0, num_classes))
    labels = np.empty(shape=0)

    for i, (x_batch, y_batch) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i == args.max:
            break

        if USE_CUDA:
            x_batch = x_batch.cuda()

        with torch.no_grad():
            batch_logits = classifier(x_batch).cpu().numpy()
        logits = np.concatenate([logits, batch_logits])
        labels = np.concatenate([labels, y_batch])

    # compute accuracy
    predicted_labels = np.argmax(logits, axis=1)
    accuracy = np.mean(predicted_labels == labels)
    print(f'Accuracy: {accuracy}')

    # save labels and logits
    data = {'split': args.split, 'labels': labels, 'logits': logits}
    np.save(os.path.join(args.root, f'{args.split}-predictions.npy'), data)


if __name__ == "__main__":
    main()
