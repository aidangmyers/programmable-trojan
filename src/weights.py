class CIFARWeights:
    def __init__(self, transform, state_dict=None, meta=None):
        self._transform = transform
        self._state_dict = state_dict
        self.meta = meta or {"categories": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]}

    def transforms(self):
        return self._transform

    def get_state_dict(self, *args, **kwargs):
        return self._state_dict
