class PaschenBackEigenvalues:
    def __init__(self):
        self.data = dict()

    def set(self, j_small, m, value):
        self.data[(j_small, m)] = value

    def __call__(self, j_small, m):
        return self.data[(j_small, m)]
