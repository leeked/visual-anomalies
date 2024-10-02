import torchvision.transforms as T

def get_transform(train, input_size=None):
    transforms = []
    if input_size:
        transforms.append(T.Resize((input_size, input_size)))
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)