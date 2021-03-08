import os
from pathlib import Path
from envyaml import EnvYAML
import random

IMAGE_EXTENSION = "png"

def get_config():
    return EnvYAML(Path(os.environ["P2MPP_DIR"]) / "configurations/common.yaml")

def get_shuffled_ls(path):
    l = os.listdir(path)
    random.shuffle(l)
    return l

def get_images(path):
    return list(filter(lambda x : x.endswith(IMAGE_EXTENSION), os.listdir(path)))

class DataListMaker():
    def __init__(self, base_path=None, ):
        if base_path is None:
            self.base_path = get_config()["shapenet_images_path"]
        else:
            self.base_path = base_path
        
        self.includes = []

    def random_seed(self, n):
        random.seed(n)

    def with_classes(self):
        return _ClassList(self.base_path)


class _ClassList():
    class _FilledClassList():
        def __init__(self, includes):
            self._includes = includes

        def with_objects(self):
            return _ObjectsList(self._includes)

    def __init__(self, base_path):
        self.base_path = base_path

    def _build_filled(self, classes):
        return _ClassList._FilledClassList(map(lambda c : Path(self.base_path) / c, classes))

    def all(self):
        return self._build_filled(filter(os.path.isdir, os.listdir(self.base_path)))

    def random(self, n):
        return self._build_filled(get_shuffled_ls(self.base_path)[:n])
        
    def named(self, classes, strict=True):
        if strict and len(set(os.listdir(self.base_path)).intersection(set(classes))) == len(classes):
            raise ValueError("Strict mode enabled. One or more of {0} doesn't exist.".format(", ".join(classes)))

        return self._build_filled(classes)


class _ObjectsList():
    class _FilledObjectsList():
        def __init__(self, objects):
            self._objects = objects

        def with_images(self):
            pass
        
    def __init__(self, classes):
        self.classes = classes
        self.objects_per_classes = {}

    def all(self):
        for c in self.classes:
            self.objects_per_classes[c] = os.listdir(c)

    def random_n(self, n):
        for c in self.classes:
            try:
                self.objects_per_classes[c] = get_shuffled_ls(c)[:n]
            except IndexError as err:
                print("Class {0} doesn't have the required {1} amount of objects.".format(c, n))
                raise err
        
    def random_perc(self, p):
        if p < 0 or p > 1:
            raise ValueError("Expected a percentage between 0 and 1, got {0}".format(p))

        for c in self.classes:
            objects_list = get_shuffled_ls(c)
            n = p // len(objects_list)
            self.objects_per_classes[c] = objects_list[:n]

    def first_n(self, n):
        for c in self.classes:
            try:
                self.objects_per_classes[c] = os.listdir(c)[:n]
            except IndexError as err:
                print("Class {0} doesn't have the required {1} amount of objects.".format(c, n))
                raise err

    def first_perc(self, p):
        if p < 0 or p > 1:
            raise ValueError("Expected a percentage between 0 and 1, got {0}".format(p))

        for c in self.classes:
            objects_list = os.listdir(c)
            n = p // len(objects_list)
            self.objects_per_classes[c] = objects_list[:n]        

    def with_images(self):
        return _ObjectsList._FilledObjectsList([Path(k) / v for k, v in self.objects_per_classes.items()])


class _ImagesList():
    class _FilledImagesList():
        def __init__(self, images):
            self.images = images
        
        def with_scenario(self):
            return _ScenarioList(self.images)

    def __init__(self, objects):
        self.objects = objects
        self.images_per_object = {}

    def all(self):
        for o in self.objects:
            self.images_per_object[o] = get_images(o)

    def first_n(self, n):
        for o in self.objects:
            try:
                self.images_per_object[o] = get_images(o)[:n]
            except IndexError as err:
                print("Object {0} doesn't have the required {1} amount of images.".format(o, n))
                raise err

    def random_n(self, n):
        for o in self.objects:
            try:
                self.images_per_object[o] = get_shuffled_ls(o)[:n]
            except IndexError as err:
                print("Object {0} doesn't have the required {1} amount of images.".format(o, n))
                raise err

    def between_range(self, n, m):
        for o in self.objects:
            k = random.randint(n, m)
            try:
                self.images_per_object[o] = get_shuffled_ls(o)[:k]
            except IndexError as err:
                print("Object {0} doesn't have the required {1} amount of images.".format(o, k))
                raise err

    def with_scenario(self):
        return _ImagesList._FilledImagesList([Path(k) / "rendering" / v for k, v in self.images_per_object.items()])
    

class _ScenarioList():
    def __init__(self, images):
        self.images = images
        self.models = []

    def keep_test(self, keep=True, path=None):
        if keep:
            return self
        
        if path is None:
            raise ValueError("Path shouldn't be 'None' when keep is False")

        test_models = set(["-".join(x.split("_")[:2]) for x in os.listdir(path)])
        ["-"i.split("/")[-4:-2] for i in self.images]


-3,-4
        

        
