import os
import torch
import numpy as np

from collections import defaultdict
from skimage import io
from torch.utils.data import Dataset
from train_utils import read_folder


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, min_images_per_class=2,
                 extensions={'.jpg', '.jpeg', '.png'}):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.label_to_indices = defaultdict(list)
        self.classes = []
        self.class_to_label = {}

        for cls_name in sorted(os.listdir(root_dir)):
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                    if os.path.splitext(f)[1].lower() in extensions]

            if len(imgs) >= min_images_per_class:
                label = len(self.classes)
                self.classes.append(cls_name)
                self.class_to_label[cls_name] = label

                for p in imgs:
                    idx = len(self.samples)
                    self.samples.append((p, label))
                    self.label_to_indices[label].append(idx)

        if len(self.classes) == 0:
            raise RuntimeError("No classes with at least min_images_per_class found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = io.imread(path)

        if self.transform:
            img = self.transform(img)

        return img, label


class ImageTripletDataset(Dataset):
    def __init__(self, root_dir, num_triplets, transform=None):
        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.transform = transform

        # read files
        self.df = df = read_folder(root_dir)

        face_classes = dict()
        for idx, label in enumerate(df['class']):
            if label not in face_classes:
                face_classes[label] = []
            face_classes[label].append((df.iloc[idx]['id'], df.iloc[idx]['ext']))

        self.triplets = self.generate_triplets(self.df, self.num_triplets, face_classes)



    def generate_triplets(self, df, num_triplets, face_classes):
        triplets = []
        classes = df['class'].unique()
        seen = set()

        while len(triplets) < num_triplets:
            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))

            ineg = np.random.randint(0, len(face_classes[neg_class]))

            anc_id = face_classes[pos_class][ianc][0]
            anc_ext = face_classes[pos_class][ianc][1]
            pos_id = face_classes[pos_class][ipos][0]
            pos_ext = face_classes[pos_class][ipos][1]
            neg_id = face_classes[neg_class][ineg][0]
            neg_ext = face_classes[neg_class][ineg][1]

            # avoid duplicates
            key = (anc_id, pos_id, neg_id)
            if key not in seen:
                seen.add(key)
                triplets.append(
                    [anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name, anc_ext, pos_ext, neg_ext]
                )

        return triplets


    def __getitem__(self, idx):
        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name, anc_ext, pos_ext, neg_ext = \
            self.triplets[idx]

        anc_img = os.path.join(self.root_dir, str(pos_name), str(anc_id) + f'{anc_ext}')
        pos_img = os.path.join(self.root_dir, str(pos_name), str(pos_id) + f'{pos_ext}')
        neg_img = os.path.join(self.root_dir, str(neg_name), str(neg_id) + f'{neg_ext}')

        anc_img = io.imread(anc_img)
        pos_img = io.imread(pos_img)
        neg_img = io.imread(neg_img)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class,
                  'neg_class': neg_class}

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample


    def __len__(self):
        return len(self.triplets)