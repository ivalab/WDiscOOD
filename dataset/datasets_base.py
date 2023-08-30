import os, sys
import torch.utils.data as data
import torchvision.datasets as dset
from PIL import Image

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)
DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))


def default_loader(path):
	return Image.open(path).convert('RGB')

def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			data = line.strip().rsplit(maxsplit=1)
			if len(data) == 2:
				impath, imlabel = data
			else:
				impath, imlabel = data[0], 0
			imlist.append( (impath, int(imlabel)) )

	return imlist

class ImageFilelist(data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root,impath))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.imlist)


class Base(data.Dataset):
    """A wrapper of datasets"""
    def __init__(self, split="test", trf=None, trf_type="test", trf_idSet="cifar100", model="resnet"):
        super().__init__()
        self.dataset = None
        self.name = None

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)
    
    def __len__(self):
        return len(self.dataset)

