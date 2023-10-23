class SiliconLayer:
    """
    Holds informations about a single layer, extracted from the detector geometry
    """

    def __init__(self, ltype, refCoord, minBound, maxBound):
        # 0 for a barrel layer, +/- 2 for positive/negative endcap layers
        self.type = ltype
        # coordinate which define the layer, from the atlas geometry --> r for barrel, z for endcap
        self.refCoord = refCoord
        # min coordinate of the layer, z for barrel, r for endcap
        self.minBound = minBound
        # max coordinate of the layer, z for barrel, r for endcap
        self.maxBound = maxBound


class DetectorModel:
    """
    Holds all the layers present in the detector.
    For now only support a detector with 4 Pixel and 4 SCT layers is implemented, without endcaps
    """

    def __init__(self):
        self.layers = None

    @staticmethod
    def buildModel_TrackML():
        """
        Build detector model type 1, only pixel and SCT layers, without endcaps.
        Geometry derived from the ACTS detector.
        """
        det = DetectorModel()
        # values from the ATLAS inner detector geometry, layerIdx 0 is the innerMost pixel layer
        # order
        det.layers = [
            SiliconLayer(0, 32., -455, 455),  # 8-2
            SiliconLayer(0, 72., -455, 455),  # 8-4
            SiliconLayer(0, 116., -455, 455),  # 8-6
            SiliconLayer(0, 172., -455, 455),  # 8-8

            SiliconLayer(0, 260., -1030, 1030),  # 13-2
            SiliconLayer(0, 360., -1030, 1030),  # 13-4
            SiliconLayer(0, 500., -1030, 1030),  # 13-6
            SiliconLayer(0, 660., -1030, 1030),  # 13-8

            SiliconLayer(0, 820., -1030, 1030),  # 17-2
            SiliconLayer(0, 1020., -1030, 1030),  # 17-4
        ]
        return det
