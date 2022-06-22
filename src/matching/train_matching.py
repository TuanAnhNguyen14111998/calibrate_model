import torch
import numpy as np

from fourier_domain_adaptation.FDA import FDA
from histogram_matching.HM import HM


class Matching():
    def __init__(self, params):
        self.matching_object = self.set_matching(params)


    def set_matching(self, params):
        if params["method"] == "fda":
            matching = FDA()
        if params["method"] == "hm":
            matching = HM()
        
        return matching
    

if __name__ == "__main__":
    Matching(
        params={
            "method": "fda"
        }
    )