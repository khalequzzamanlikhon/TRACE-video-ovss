"""Text prompts and frozen class-name lists (identical for every method)."""

# 80 OpenAI ImageNet templates (as used by SCLIP / ProxyCLIP / NACLIP).
IMAGENET_TEMPLATES = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.',
    'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
    'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.',
    'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.',
    'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.',
    'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.',
    'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
    'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.',
    'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.',
    'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.',
    'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.',
    'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.',
    'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.',
    'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
]
assert len(IMAGENET_TEMPLATES) == 80

# Cityscapes-19 / KITTI-STEP (trainId order). Plain names, one query per class.
CITYSCAPES_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                      'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                      'motorcycle', 'bicycle']
CITYSCAPES_QUERIES = [[c] for c in CITYSCAPES_CLASSES]

# CamVid-11 (SegNet index order, void = 11). Synonyms frozen before any test evaluation.
CAMVID_CLASSES = ['sky', 'building', 'pole', 'road', 'sidewalk', 'tree', 'sign symbol', 'fence', 'car',
                  'pedestrian', 'bicyclist']
CAMVID_QUERIES = [['sky'], ['building'], ['pole'], ['road'], ['sidewalk', 'pavement'], ['tree', 'vegetation'],
                  ['traffic sign'], ['fence'], ['car'], ['pedestrian', 'person'], ['bicyclist', 'cyclist', 'bicycle']]
