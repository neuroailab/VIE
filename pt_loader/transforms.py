import math
import numbers
import random
from PIL import Image, ImageOps

import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import Lambda
from torchvision.transforms import Compose

COLOR_MEAN = [0.485, 0.456, 0.406]
COLOR_STD = [0.229, 0.224, 0.225]


def video_transform_rdsz(
        crop_size=224):
    """Return prepared video transform.

    Args:
        crop_size (int, optional): Defaults to 224. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    return Compose([
        GroupRandomSizedCrop(crop_size),
        GroupRandomHorizontalFlip(),
        ByteStack(),
    ])


def video_transform_color(
        frame_size_min=256, frame_size_max=320,
        crop_size=224):
    """Return prepared video transform.

    Args:
        frame_size_min (int, optional): Defaults to 256. Min frame size before cropping.
        frame_size_max (int, optional): Defaults to 320. Max frame size before cropping.
        crop_size (int, optional): Defaults to 224. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    return Compose([
        RandomGroupResize(frame_size_min, frame_size_max),
        GroupRandomCrop(crop_size),
        GroupColorJitter(),
        GroupRandomHorizontalFlip(),
        ByteStack(),
    ])


def video_transform_multiscalecrop(
        scales=[1, 0.84089641525, 0.7071067811803005, 0.5946035574934808, 0.5],
        crop_positions=['c', 'tl', 'tr', 'bl', 'br'],
        size=112):
    """The preprocessing pipeline used in 3D ResNet paper.
    1. Randomly select 1 position from 4 corners or center
    2. Randomly select 1 crop scale from 5 options ranging from 0.5 to 1
    3. Resize to 112*112
    4. Horizontally flip with 50% probability

    Args:
        scales (list of float, optional): Use the same 5 scales as in 3D resnet paper.
        crop_size (int, optional): Defaults to 112. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    return Compose([
        GroupRandomCornerCrop(scales=scales, crop_positions=crop_positions, size=size),
        GroupRandomHorizontalFlip(),
        ByteStack(),
    ])



class GroupRandomCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, scales, crop_positions, size):
        self.scales = scales
        self.crop_positions = crop_positions
        self.size = size

    def __call__(self, frames):

        # Choose a random position and random scale
        scale = self.scales[random.randint(0, len(self.scales) - 1)]
        crop_position = self.crop_positions[random.randint(0, len(self.crop_positions) - 1)]

        # Get the crop size for the scale cropping
        image_width = frames[0].size[0]
        image_height = frames[0].size[1]

        min_length = min(image_width, image_height)
        crop_size = int(min_length * scale)

        # Do the scale cropping at the chosen position
        if crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height
        else:
            raise ValueError("Crop position must be 1 of c, tl, tr, bl, br")

        out_frames = []        
        for img in frames:
            assert img.size[0]==image_width and img.size[1]==image_height
            # Multi-scale cropping
            img = img.crop((x1, y1, x2, y2))
            # Resize to the input size
            img = img.resize((self.size, self.size), Image.BILINEAR)
            out_frames.append(img)
        
        return out_frames





def video_transform_color_rdsz(
        crop_size=224):
    """Return prepared video transform.

    Args:
        crop_size (int, optional): Defaults to 224. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    return Compose([
        GroupRandomSizedCrop(crop_size),
        GroupRandomHorizontalFlip(),
        GroupColorJitter(),
        ByteStack(),
    ])


def video_transform(frame_size_min=256, frame_size_max=320,
                    crop_size=224):
    """Return prepared video transform.

    Args:
        frame_size_min (int, optional): Defaults to 256. Min frame size before cropping.
        frame_size_max (int, optional): Defaults to 320. Max frame size before cropping.
        crop_size (int, optional): Defaults to 224. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    return Compose([
        RandomGroupResize(frame_size_min, frame_size_max),
        GroupRandomCrop(crop_size),
        GroupRandomHorizontalFlip(),
        ByteStack(),
    ])


def video_transform_val(
        dataset="kinetics",
        frame_size=256,
        crop_size=224):
    """Return prepared video transform.

    Args:
        frame_size (int, optional): Defaults to 256. Frame size before cropping.
        crop_size (int, optional): Defaults to 224. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    # Vertically flip infant_headcam frames
    if dataset == "infant":
        return Compose([
            GroupResize(frame_size),
            GroupCenterCrop(crop_size),
            VerticalFlip(),
            ByteStack(),
        ])

    return Compose([
        GroupResize(frame_size),
        GroupCenterCrop(crop_size),
        ByteStack(),
    ])


def video_OPN_transform_color(
        frame_size_min=256, frame_size_max=320,
        crop_size=80):
    """Return prepared video transform.

    Args:
        frame_size_min (int, optional): Defaults to 256. Min frame size before cropping.
        frame_size_max (int, optional): Defaults to 320. Max frame size before cropping.
        crop_size (int, optional): Defaults to 80. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    return Compose([
        RandomGroupResize(frame_size_min, frame_size_max),
        GroupRandomCrop(crop_size + 20),
        SpatialJitter(crop_size),
        GroupColorJitter(),
        GroupRandomHorizontalFlip(),
        ByteStack(),
    ])


def video_OPN_transform_sep_color(
        frame_size_min=256, frame_size_max=320,
        crop_size=80):
    """Return prepared video transform.

    Args:
        frame_size_min (int, optional): Defaults to 256. Min frame size before cropping.
        frame_size_max (int, optional): Defaults to 320. Max frame size before cropping.
        crop_size (int, optional): Defaults to 80. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    return Compose([
        RandomGroupResize(frame_size_min, frame_size_max),
        GroupRandomCrop(crop_size + 20),
        SpatialJitter(crop_size),
        SeparateColorJitter(),
        GroupRandomHorizontalFlip(),
        ByteStack(),
    ])


def video_3DRot_transform(
        frame_size_min=136, frame_size_max=137,
        crop_size=112):
    """Return prepared video transform.

    Args:
        frame_size_min (int, optional): Defaults to 256. Min frame size before cropping.
        frame_size_max (int, optional): Defaults to 320. Max frame size before cropping.
        crop_size (int, optional): Defaults to 80. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    return Compose([
        RandomGroupResize(frame_size_min, frame_size_max),
        GroupRandomCrop(crop_size),
        GroupRandomHorizontalFlip(),
        GroupRotation(),
        ByteStack(),
    ])


def video_3DRot_transform_real_resize(
        frame_size=(136, 136),
        crop_size=112):
    return Compose([
        GroupResize(frame_size),
        GroupRandomCrop(crop_size),
        GroupRandomHorizontalFlip(),
        GroupRotation(),
        ByteStack(),
    ])


def video_3DRot_transform_val(
        frame_size=136,
        crop_size=112):
    """Return prepared video transform.

    Args:
        frame_size (int, optional): Defaults to 256. Frame size before cropping.
        crop_size (int, optional): Defaults to 224. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    return Compose([
        GroupResize(frame_size),
        GroupCenterCrop(crop_size),
        GroupRotation(),
        ByteStack(),
    ])


def video_3DRot_finetune_val(
        frame_size=136,
        crop_size=112):
    """Return prepared video transform.

    Args:
        frame_size (int, optional): Defaults to 256. Frame size before cropping.
        crop_size (int, optional): Defaults to 224. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    return Compose([
        GroupResize(frame_size),
        GroupCenterCrop(crop_size),
        ByteStack(),
    ])


def video_3DRot_finetune_transform(
        frame_size=(136, 136),
        crop_size=112):
    """Return prepared video transform.

    Args:
        frame_size_min (int, optional): Defaults to 256. Min frame size before cropping.
        frame_size_max (int, optional): Defaults to 320. Max frame size before cropping.
        crop_size (int, optional): Defaults to 80. Frame size after cropping (final input size).

    Returns:
        [torchvision.Transform]: Video
    """

    return Compose([
        GroupResize(frame_size),
        GroupRandomCrop(crop_size),
        GroupRandomHorizontalFlip(),
        ByteStack(),
    ])


class SpatialJitter(object):
    def __init__(self, crop_size=80, jitter_dis=5):
        self.jitter_dis = jitter_dis
        self.crop_size = crop_size

    def __call__(self, frames):

        w, h = frames[0].size

        crop_size = self.crop_size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)

        out_frames = []
        for img in frames:
            assert(img.size[0] == w and img.size[1] == h)
            sx = random.randint(-self.jitter_dis, self.jitter_dis)
            sy = random.randint(-self.jitter_dis, self.jitter_dis)
            if x1 + sx > 0 and x1 + crop_size + sx < w:
                newx = x1 + sx
            else:
                newx = x1
            if y1 + sy > 0 and y1 + crop_size + sy < h:
                newy = y1 + sy
            else:
                newy = y1
            out_frames.append(img.crop((
                newx, newy, newx + crop_size, newy + crop_size)))
        return out_frames


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, frames):

        w, h = frames[0].size
        th, tw = self.size

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        out_frames = []
        for img in frames:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_frames.append(img)
            else:
                out_frames.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_frames


class GroupCenterCrop(object):
    def __init__(self, size):
        self.transform = torchvision.transforms.CenterCrop(size)

    def __call__(self, frames):
        return [self.transform(img) for img in frames]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of prob."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, frames):
        if random.random() < self.prob:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in frames]
        else:
            return frames


class VerticalFlip(object):
    """Vertically flips the given PIL.Image."""

    def __call__(self, frames):
        return [img.transpose(Image.FLIP_TOP_BOTTOM) for img in frames]
        

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean, std)

    def __call__(self, tensor):
        return torch.stack([self.normalize(f) for f in tensor])


class GroupResize(object):
    """Rescales the input PIL.Image to the given 'size'.

    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.transform = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, frames):
        return [self.transform(img) for img in frames]


class GroupAnyRotation(object):
    """Rotate the video by angle.

    Args:
        angles (sequence or float or int): Angle to select from.
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, angles=[0, 90, 180, 270],
                 resample=False, expand=False, center=None):
        self.angles = angles
        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, frames):
        """
        Args:
            frames(sequence of PIL Image): frames to be rotated.

        Returns:
            Sequence of PIL Image: Rotated frames.
        """
        return [F.rotate(f, ang, self.resample, self.expand, self.center)
                for f in frames for ang in self.angles]

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(angles={0}'.format(self.angles)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class GroupRotation(object):
    """Rotate the video.
    """

    def __init__(self,
                 ops=[Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270],
                 ):
        self.ops = ops

    def __call__(self, frames):
        """
        Args:
            frames(sequence of PIL Image): frames to be rotated.

        Returns:
            Sequence of PIL Image: Rotated frames.
        """
        return frames + [f.transpose(op) for f in frames for op in self.ops]

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(angles={0}'.format(self.angles)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomGroupResize(object):
    """Randomly rescales the input PIL.Image between the given 'size' range.

    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
            self, size_min, size_max,
            size_interval=1, interpolation=Image.BILINEAR):
        self.transforms = [
            torchvision.transforms.Resize(size_now, interpolation)
            for size_now in range(size_min, size_max, size_interval)]
        self.no_trans = len(self.transforms)

    def __call__(self, frames):
        idx_trans_now = random.randint(0, self.no_trans - 1)
        return [self.transforms[idx_trans_now](img) for img in frames]


class Stack(object):
    """Stack PIL images as torch tensor images along specified dim."""

    def __init__(self, dim=0):
        self.dim = dim
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, frames):
        return torch.stack([self.to_tensor(f) for f in frames], self.dim)


def to_byte_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)

    img = img.view(pic.size[1], pic.size[0], nchannel)
    return img


class ByteStack(object):
    """Stack PIL images as normalized torch tensor images along first dim."""

    def __init__(self):
        """Normalized and Stack PIL images.

        Args:
            mean ([List]): Means for each channel dim.
            std ([List]): Stdevs for each channel dim.
            permute_dims (bool, optional): Defaults to True. Permute dims 0 and 1 of stacked tensor.
        """

        self.transform = Compose([
            to_byte_tensor,
        ])

    def __call__(self, frames):
        out = torch.stack([self.transform(f) for f in frames])
        return out


class NormalizedStack(object):
    """Stack PIL images as normalized torch tensor images along first dim."""

    def __init__(self, mean, std, permute_dims=True):
        """Normalized and Stack PIL images.

        Args:
            mean ([List]): Means for each channel dim.
            std ([List]): Stdevs for each channel dim.
            permute_dims (bool, optional): Defaults to True. Permute dims 0 and 1 of stacked tensor.
        """

        self.permute_dims = permute_dims
        self.transform = Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])

    def permute(self, frame):
        return frame.permute(1, 0, 2, 3)

    def __call__(self, frames):
        out = torch.stack([self.transform(f) for f in frames])
        if self.permute_dims:
            out = self.permute(out)
        return out


class IdentityTransform(object):

    def __call__(self, data):
        return data


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None,
                 sample_flips=False, more_fix_crop=False):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)
        self.more_fix_crop = more_fix_crop
        self.sample_flips = sample_flips

        if scale_size is not None:
            self.scale_worker = GroupResize(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(self.more_fix_crop,
                                                      image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            if self.sample_flips:
                oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1,
                 fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [
            input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop(
                (offset_w,
                 offset_h,
                 offset_w +
                 crop_w,
                 offset_h +
                 crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(
                x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [
            self.input_size[0] if abs(
                x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) / 4
        h_step = (image_h - crop_h) / 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Randomly crop the given PIL.Image.

    Each image is cropeed to a random size of (0.08 to 1.0) of the original size
    and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(
                    img.resize(
                        (self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupResize(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class GroupColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image

    See torchvision.transforms.ColorJitter
    Add a random grayscale
    """

    def __init__(
            self,
            brightness=0.4, contrast=0.4,
            saturation=0.4, hue=0.4,
            grayscale=0.3):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.grayscale = grayscale

    def _get_transform(self):
        curr_brightness = random.uniform(
            1 - self.brightness, 1 + self.brightness)
        curr_contrast = random.uniform(1 - self.contrast, 1 + self.contrast)
        curr_saturation = random.uniform(
            1 - self.saturation, 1 + self.saturation)
        curr_hue = random.uniform(-self.hue, self.hue)

        transforms = []
        transforms.append(
            Lambda(lambda img: F.adjust_brightness(img, curr_brightness)))
        transforms.append(
            Lambda(lambda img: F.adjust_contrast(img, curr_contrast)))
        transforms.append(
            Lambda(lambda img: F.adjust_saturation(img, curr_saturation)))
        transforms.append(
            Lambda(lambda img: F.adjust_hue(img, curr_hue)))
        random.shuffle(transforms)
        curr_grayscale = random.uniform(0, 1)
        if curr_grayscale < self.grayscale:
            transforms.append(
                Lambda(lambda img: F.to_grayscale(
                    img, num_output_channels=3)))
        transform = Compose(transforms)
        return transform

    def __call__(self, img_group):
        transform = self._get_transform()

        out_group = list()
        for img in img_group:
            out_group.append(transform(img))
        return out_group


class SeparateColorJitter(GroupColorJitter):
    def __call__(self, img_group):
        out_group = list()
        for img in img_group:
            transform = self._get_transform()
            out_group.append(transform(img))
        return out_group
