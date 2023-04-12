import numpy as np
from .storage import *


def doublet_making(constants, spStorage: SpacepointStorage, detModel, doubletsStorage: DoubletStorage):
    """
    Implementation of the DoubletCountingKernelCuda.cuh
    """
    for sliceIdx in range(constants.nPhiSlices):  # iterate for each phi slice
        for layerIdx in range(constants.nLayers):  # iterate for each layer
            slr: SpacepointLayerRange = spStorage.phiSlices[sliceIdx]
            spBegin = slr.layerBegin[layerIdx]
            spEnd = slr.layerEnd[layerIdx]
            nMiddleSPs = spEnd - spBegin
            if nMiddleSPs == 0:
                continue  # break

            for spmIdx in range(spBegin, spEnd):  # iterate over the spacepoint of the phiSlice/layer
                isPixel = spStorage.type[spmIdx]
                spmZ = spStorage.z[spmIdx]
                spmR = spStorage.r[spmIdx]
                spmHid = spStorage.idsp[spmIdx]  # TODO debug

                inner = []
                outer = []

                for deltaSlice in range(-1, 2):  # iterate over adjacent and current phi slices
                    nextSlice = sliceIdx + deltaSlice
                    if nextSlice >= constants.nPhiSlices:
                        nextSlice = 0
                    if nextSlice < 0:
                        nextSlice = constants.nPhiSlices - 1
                    next_slr = spStorage.phiSlices[nextSlice]

                    for next_layer in range(0, constants.nLayers):
                        # iterate over each layer of the adjacents /current layers

                        next_spBegin = next_slr.layerBegin[next_layer]
                        next_spEnd = next_slr.layerEnd[next_layer]
                        if next_spBegin == next_spEnd:  # no spacepoint --> next
                            continue

                        if next_layer == layerIdx:  # if same layer --> next
                            if deltaSlice != 0:  # only when same layer AND same angle
                                continue
                            # look for very short radius (>10)
                            delta_radius = spStorage.r[next_spBegin:next_spEnd] - spmR
                            mask_rad = np.abs(delta_radius) < 10
                            # look for theta angles not too big (we want vertical-ish segments)
                            delta_radius[delta_radius == 0] = 0.1  # avoid nan
                            thetas = (spStorage.z[next_spBegin:next_spEnd] - spmZ) / delta_radius
                            mask_theta = np.abs(thetas) < constants.maxCtg
                            # get all corresponding hits
                            final_mask = (mask_theta == 1) & (mask_rad == 1)
                            all_ids = np.arange(next_spBegin, next_spEnd)[final_mask]
                            # finally, create the segments
                            for spIdx in all_ids:
                                if spIdx == spmIdx: continue
                                if spStorage.r[spIdx] - spmR > 0:
                                    outer.append(spIdx)
                                else:
                                    inner.append(spIdx)
                            continue

                        # we are looking at all the layer, so here we ensure the doublet
                        # is not too long
                        layerGeo = detModel.layers[next_layer]
                        isBarrel = layerGeo.type == 0
                        refCoord = layerGeo.refCoord
                        if isBarrel and np.abs(
                                refCoord - spmR) > constants.maxDoubletLength:  # if double is too long -->next
                            continue

                        # compute min/max boundaries
                        minCoord = 10000.0
                        maxCoord = -10000.0
                        if isBarrel:
                            # see schema p 57 (Julien) - Figure 3.5
                            # projection on the current layer:
                            # tan(theta) = refc/x = spmR/(spmZ-zMinus) => x = (smpZ-zMinus)/smpR * refc
                            minCoord = constants.zMinus + refCoord * (spmZ - constants.zMinus) / spmR
                            maxCoord = constants.zPlus + refCoord * (spmZ - constants.zPlus) / spmR
                        else:
                            minCoord = spmR * (refCoord - constants.zMinus) / (spmZ - constants.zMinus)
                            maxCoord = spmR * (refCoord - constants.zPlus) / (spmZ - constants.zPlus)

                        if minCoord > maxCoord:
                            tmp = minCoord
                            minCoord = maxCoord
                            maxCoord = tmp

                        # TODO: and ? if we have a min outside the layer but max inside, we could
                        # still have interesting hits...
                        if layerGeo.maxBound < minCoord or layerGeo.minBound > maxCoord:
                            continue

                        # we computed the limit of the zone that is interesting. Now, we actually look at the hits.
                        for spIdx in range(next_spBegin,
                                           next_spEnd):  # iterate over spacepoints in the adjacent / same phi bins and outer/inner layers
                            zsp = spStorage.z[spIdx]
                            rsp = spStorage.r[spIdx]
                            spHid = spStorage.idsp[spIdx]  # TODO debug
                            spCoord = zsp if isBarrel else rsp
                            if spCoord < minCoord or spCoord > maxCoord:  # if out of boundaries --> next
                                continue

                            isPixel2 = spStorage.type[spIdx]
                            delta_radius = rsp - spmR  # delta_radius
                            if np.abs(delta_radius) > constants.maxDoubletLength or np.abs(
                                    delta_radius) < constants.minDoubletLength:
                                continue
                            if (not constants.doPSS) and delta_radius < 0 and (not isPixel) and isPixel2:
                                # TODO: what happens if we switch doPSS to true ? not clear...
                                continue

                            delta_z = zsp - spmZ
                            t = delta_z / delta_radius  # which is very close to the delta theta ??
                            if np.abs(t) > constants.maxCtg:  # segments too horizontal
                                continue
                            # projecting the 2nd point to the outer radius and see the z.
                            # ... redondant with ctg ?
                            # outZ = zsp + (constants.maxOuterRadius - rsp) * t
                            outZ = constants.maxOuterRadius * t  # easier
                            if outZ < constants.minOuterZ or outZ > constants.maxOuterZ:
                                assert not (constants.minOuterZ <= outZ <= constants.maxOuterRadius)
                                continue
                            if delta_radius > 0:
                                outer.append(spIdx)
                            else:
                                inner.append(spIdx)
                nInner = len(inner)
                nOuter = len(outer)
                is_good = nInner > 0 or nOuter > 0

                if is_good:
                    doubletsStorage.spmIdx.append(spmIdx)

                    doubletsStorage.innerStart.append(len(doubletsStorage.inner))
                    doubletsStorage.inner += inner
                    doubletsStorage.nI += nInner

                    doubletsStorage.outerStart.append(len(doubletsStorage.outer))
                    doubletsStorage.outer += outer
                    doubletsStorage.nO += nOuter
