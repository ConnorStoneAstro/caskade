class ActiveContext:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        self.module.active = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.active = False
