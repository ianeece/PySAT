import numpy as np
from pysptools.eea import PPI, FIPPI, NFINDR, ATGP

def emi(data, col = 'wvl', emi_method = 'FIPPI', n_endmembers = 6):
    supported_methods = ("FIPPI", "PPI", "N-FINDR", "ATGP")
    try:
        if emi_method.upper() in supported_methods:
            if emi_method == 'PPI':
                method = PPI()
            if emi_method == 'FIPPI':
                method = FIPPI()
            if emi_method == 'N-FINDR':
                method = NFINDR()
            if emi_method == 'ATGP':
                method = ATGP()

        else:
            print(f"{emi_method} is not a supported method.  Supported methods are {supported_methods}")
            return 1
    except KeyError:
        print(f"Unable to instantiate class from {emi_method}.")
        return 1

    spectra = data[col].to_numpy()
    if len(spectra.shape) == 2:
        spectra = np.expand_dims(spectra, 0)

    method.extract(spectra, n_endmembers)
    endmember_indices = [i[0] for i in method.get_idx()]
    indices = np.zeros(spectra.shape[1], dtype=int)
    indices[endmember_indices] = 1
    data[("endmembers", emi_method)] = indices
    return data, endmember_indices
