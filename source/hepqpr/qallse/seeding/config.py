import numpy as np


class SeedingConfig:

    def __init__(self, nLayers=10):
        self.nLayers = nLayers
        self.nPhiSlices = 53
        self.zTolerance = 3.0
        self.maxEta = 2.7
        self.maxDoubletLength = 300.0  # 200.0 # LL: longer, since we added a volume !
        self.minDoubletLength = 10.0
        self.maxOuterRadius = 550.0
        self.doPSS = False
        self.zMinus = -350
        self.zPlus = 350
        self._compute_derived_attrs()

    def _compute_derived_attrs(self):
        self.maxTheta = 2 * np.arctan(np.exp(-self.maxEta))
        self.maxCtg = np.cos(self.maxTheta) / np.sin(self.maxTheta)
        self.minOuterZ = self.zMinus - self.maxOuterRadius * self.maxCtg - self.zTolerance
        self.maxOuterZ = self.zPlus + self.maxOuterRadius * self.maxCtg + self.zTolerance


class HptSeedingConfig(SeedingConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zMinus, self.zPlus = -150, 150
        self._compute_derived_attrs()
