from importlib import import_module


def split_module_name(abs_class_name):
    """
    It takes a fully qualified class name (e.g. `"foo.bar.Baz"`) and returns a tuple of the module path (e.g. `"foo.bar"`)
    and the class name (e.g. `"Baz"`)

    :param abs_class_name: The absolute name of the class
    :return: The absolute module path and the class name.
    """
    abs_module_path = ".".join(abs_class_name.split(".")[:-1])
    class_name = abs_class_name.split(".")[-1]
    return abs_module_path, class_name


def dynamic_import(abs_module_path, class_name):
    """
    It dynamically imports a class from a module

    :param abs_module_path: The absolute path to the module you want to import
    :param class_name: The name of the class you want to instantiate
    :return: The class object
    """
    module_object = import_module(abs_module_path)
    target_class = getattr(module_object, class_name)
    return target_class


def module_import(path):
    """
    It takes a string like `"foo.bar.baz"` and returns the module object `baz` from the package `foo.bar`

    :param path: The path to the module you want to import
    :return: The module object.
    """
    return dynamic_import(*split_module_name(path))
