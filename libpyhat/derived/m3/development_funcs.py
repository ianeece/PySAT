def h2o2_func(bands, _):
    R1348, R1408, R1428, R1448, R1578 = bands

    RC = (R1348 + R1578) / 2
    LC = (R1428 + R1448) / 2
    BB = R1408
    return 1 - 2 * (BB / (RC + LC))

def h2o3_func(bands, _):
    R1428, R1448, R1488, R1508, R1528 = bands

    RC = (R1428 + R1448) / 2
    LC = (R1508 + R1528) / 2
    BB = R1488
    return 1 - 2 * (BB / (RC + LC))

def h2o4_func(bands, _):
    R2218, R2258, R2378, R2418, R2298, R2338 = bands

    RC = (R2218 + R2258) / 2
    LC = (R2378 + R2418) / 2
    BB = (R2298 + R2338) / 2
    return 1 - 2 * (BB / (RC + LC))

def h2o5_func(bands, _):
    R2578, R2618, R2658, R2698, R2738 = bands

    RC = (R2578 + R2618 + R2658) / 3
    BB = (R2698 + R2738) / 2
    return 1 - (BB / RC)

def ice_func(bands, _):
    R2538, R2578, R2618, R2817, R2857, R2897 = bands

    RC = (R2538 + R2578 + R2618) / 3
    BB = (R2817 + R2857 + R2897) / 3
    return 1 - (BB / RC)
