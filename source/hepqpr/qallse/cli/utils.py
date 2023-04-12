# ======= instantiation

def _to_camelcase(text):
    """
    Converts underscore_delimited_text to CamelCase.
    Example: "tool_name" becomes "ToolName"
    """
    return ''.join(word.title() for word in text.split('_'))


def qallse_class_from_string(cls):
    try:
        import importlib
        if cls.startswith('.'):
            module = 'hepqpr.qallse' + cls
            cls = cls[1:]
        else:
            module = '.'.join(cls.split('.')[:-1])
            cls = cls.split('.')[-1]
        cls = _to_camelcase(cls)
        return getattr(importlib.import_module(module), cls)
    except Exception as err:
        raise RuntimeError(f'Error instantiating "{module}.{cls}". Are you sure it exists ?') from err


def extra_to_dict(extra, typ=str):
    # parse any extra argument to pass to the model's constructor
    dict_extra = dict()
    for s in extra:
        try:
            k, v = s.split('=')
            dict_extra[k.strip()] = typ(v.strip())
        except:
            print(f'error: {s} could not be processed. Extra args should be in the form k=<v:{typ.__name__}>')
    return dict_extra
