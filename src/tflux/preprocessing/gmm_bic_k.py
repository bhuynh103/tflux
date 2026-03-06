from sklearn.mixture import GaussianMixture

def gmm_bic_k(normals, k_range=(2,6)):
    bics = {}
    for k in range(k_range[0], k_range[1] + 1):
        gm = GaussianMixture(n_components=k, covariance_type='full', n_init=5)
        gm.fit(normals)
        bics[k] = gm.bic(normals)
    return min(bics, key=bics.get), bics


def extract_junctions():
    return