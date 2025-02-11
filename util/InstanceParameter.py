import numpy as np

FLEX = [2, 2, 3, 3, 2, 2, 2, 2, 5]
GREP = [3, 3, 4, 6, 8, 4, 3, 2, 5]
GZIP = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]
MAKE = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
NANOXML = [2, 2, 2, 2, 2, 4, 2]
SED = [2, 2, 2, 2, 6, 10, 2, 4, 2, 2, 3]
SIM_1 = [2,2,2,2,2]
SIM_2 = [2,2,2,2,2,2,2,2]
SIM_3 = [3,3,3,3,3]
SIM_4 = [3,3,3,3,3,3,3]
SIM_5 = [3,3,3,3,3,3,3,3]
SIM_6 = [4,4,4,4,4]
SIM_8 = [3,3,3,3,3,3]


prog_param = {
    "inst1":FLEX,
    "inst2":GREP,
    "inst3":GZIP,
    "inst4": SED,
    "inst5":NANOXML,
    'inst6':SIM_1,
    'inst7':SIM_2,
    "inst8": MAKE,
    'inst9':SIM_3,
    'inst10':SIM_4,
    'inst11':SIM_5,
    'inst12':SIM_6,
}

prog_step={
    "inst1":500000,
    "inst2":1000000,
    "inst3":500000,
    "inst4": 500000,
    "inst5":500000,
    'inst6':500000,
    'inst7':500000,
    "inst8": 500000,
    'inst9':1000000,
    'inst10':1000000,
    'inst11':500000,
    'inst12':1000000,
}

def get_TL(param):
    tl = [1]
    for i in range(-1, -len(param), -1):
        tl.insert(0, np.prod(param[i:]))
    return tl

