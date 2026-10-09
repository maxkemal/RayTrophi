#ifndef RT_MAC_SOLID_WEIGHT
#define RT_MAC_SOLID_WEIGHT

// Caller declares uw/vw/ww and includes sim_mac_lane.glsl. Solid openness is
// distinct from the P2G accumulation weight. Missing faces are fully open;
// domain wall policy is applied by fw_* / faceWeight before this lookup.
float macSolidWeight(int component, int i, int j, int k) {
    uint address = macAddress(component, i, j, k);
    if (address == MAC_ABSENT) {
        return 1.0;
    }
    return component == 0 ? uw[address] : (component == 1 ? vw[address] : ww[address]);
}

#endif
