from collections import namedtuple

KFoldDto = namedtuple('KFoldDto', 'trainFiles kFoldNumber')

def convert_input_to(class_, request):
    def wrap(f):
        def decorator(*args):
            obj = class_(**request.get_json())
            return f(obj)
        return decorator
    return wrap
