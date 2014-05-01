def require_type(obj, clazz):
    if not isinstance(obj, clazz):
            raise ValueError("invalid type - expected: {0}, got: {1}".format(clazz, object.__class__))
