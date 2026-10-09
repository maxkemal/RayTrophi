// Must match FluidGpuDispatch::groups256. Reject padded WORKGROUPS before
// multiplying by 256: a padded rectangle near UINT_MAX can otherwise wrap its
// lane back to zero and process the first particles/faces twice.
uint simLane256(uint count) {
    uint logical_groups = count / 256u + (count % 256u != 0u ? 1u : 0u);
    uint group = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    if (group >= logical_groups) {
        return 0xffffffffu;
    }
    return group * 256u + gl_LocalInvocationID.x;
}
