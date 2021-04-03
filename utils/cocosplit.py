import json
import funcy
from sklearn.model_selection import train_test_split


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump(
            {
                'info': info,
                'licenses': licenses,
                'images': images,
                'annotations': annotations,
                'categories': categories
            },
            coco,
            indent=2,
            sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids,
                         annotations)


def split(annotation_file,
          train_file,
          test_file,
          split=0.8,
          having_annotations=True):
    with open(annotation_file, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco.get('licenses', [])
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']),
                                             annotations)

        if having_annotations:
            images = funcy.lremove(
                lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=split)

        save_coco(train_file, info, licenses, x,
                  filter_annotations(annotations, x), categories)
        save_coco(test_file, info, licenses, y,
                  filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(
            len(x), train_file, len(y), test_file))
