

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

// ═══════════════════════════════════════════════════════════════════════════
// FOLIAGE DEFORMATION KERNEL
// ═══════════════════════════════════════════════════════════════════════════

extern "C" __global__ void deform_foliage_kernel(
    const float3* __restrict__ rest_positions, 
    float3* __restrict__ output_positions,        
    int vertex_count,                         
    float3 wind_direction,                    
    float wind_strength,                      
    float wind_speed,                         
    float time,                               
    float mesh_height,                        
    float3 mesh_pivot,
    const float3* __restrict__ colors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertex_count) return;

    // Load rest position
    float3 pos = rest_positions[idx];

    // Early exit if no wind (optimization)
    if (wind_strength < 0.001f) {
        output_positions[idx] = pos;
        return;
    }

    // 1. Calculate Height Factor (0 at base, 1 at top)
    // We assume Y-up coordinate system for foliage
    float local_y = pos.y - mesh_pivot.y;
    
    // Clamp height factor 0-1
    // Avoid division by zero
    float h_denom = (mesh_height > 0.001f) ? mesh_height : 1.0f;
    float normalized_height = fmaxf(0.0f, fminf(1.0f, local_y / h_denom));
    
    // -----------------------------------------------------------------------
    // TUNING: Trunk Protection & Curve
    // -----------------------------------------------------------------------
    // "Trunk" logic: Bottom 20% is completely rigid (no bend)
    float trunk_threshold = 0.2f; 
    float height_factor = 0.0f;
    
    if (normalized_height > trunk_threshold) {
        // Remap (0.2 -> 1.0) to (0.0 -> 1.0) for the branching part
        float t = (normalized_height - trunk_threshold) / (1.0f - trunk_threshold);
        
        // Cubic curve for branching part (stiff start, flexible tip)
        height_factor = t * t * t;
    } else {
        height_factor = 0.0f; // Rigid trunk
    }
    
    // -----------------------------------------------------------------------
    // VERTEX COLOR MASKS (If available)
    // -----------------------------------------------------------------------
    float color_stiffness = 0.0f; // 0 = Flexible, 1 = Rigid
    float flutter_mask = 0.0f;    // 0 = Branch, 1 = Leaf
    
    if (colors) {
        float3 c = colors[idx];
        color_stiffness = c.x; // Red Channel = Stiffness
        flutter_mask = c.z;    // Blue Channel = Leaf Detail
    }
    
    // Apply stiffness from vertex color (Multiplicative mask)
    // If Red=1, branch is rigid. If Red=0, it bends fully.
    height_factor *= (1.0f - color_stiffness);

    // 2. Multi-Frequency Oscillation
    // ... (Use previous frequency logic) ...
    float phase = (pos.x * wind_direction.x + pos.z * wind_direction.z) * 0.05f;
    float wave_primary = sinf(phase + time * wind_speed) * 1.0f;
    float wave_secondary = sinf(phase * 2.3f + time * wind_speed * 1.7f) * 0.35f;
    float wave_tertiary = sinf(phase * 4.7f + time * wind_speed * 2.9f) * 0.15f;
    float oscillation = (wave_primary + wave_secondary + wave_tertiary) / 1.5f;
    
    // 3. Calculate Displacement
    // Scale by strength and height
    // Multiplier increased to 3.5 to make upper branches bend significantly despite trunk stiffness
    float displacement_amount = oscillation * wind_strength * height_factor * 3.5f; 
    
    float3 displacement;
    displacement.x = wind_direction.x * displacement_amount;
    
    // Y-Compression: Simulate arcing (tip drops slightly when bent)
    // Helps avoid "Shearing" look
    displacement.y = -fabsf(displacement_amount) * 0.2f * normalized_height; 
    
    displacement.z = wind_direction.z * displacement_amount;
    
    // -----------------------------------------------------------------------
    // LEAF FLUTTER (Micro-movement for leaves)
    // -----------------------------------------------------------------------
    if (flutter_mask > 0.05f) {
        // High frequency chaos
        float f_speed = wind_speed * 4.0f;
        float f_phase = idx * 0.1f; // Randomize per vertex
        float flutter = sinf(time * f_speed + f_phase) * 0.03f * wind_strength * flutter_mask;
        
        displacement.x += flutter;
        displacement.y += flutter * 0.5f;
        displacement.z += flutter;
    }

    // 4. Apply to Vertex
    // Add displacement to rest position
    output_positions[idx] = make_float3(pos.x + displacement.x, pos.y + displacement.y, pos.z + displacement.z);
}

// Wrapper function to launch kernel from C++
// Wrapper function removed - Kernel launched via CUDA Driver API (cuLaunchKernel)
