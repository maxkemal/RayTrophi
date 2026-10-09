    if (buffers.matter_runtime) {
        auto runtime = std::move(buffers.matter_runtime);
        Fluid::releaseMatterGrainGpu(compute, runtime->grain);
        Fluid::releaseMatterGrainFluidGpuStorage(compute, runtime->liquid_coupling);
        releaseGridDomainComputeBuffers(compute, runtime->granular);
        Fluid::destroyMatterGpuPartition(compute, runtime->partition);
        compute.destroyBuffer(runtime->rest_mass);
        compute.destroyBuffer(runtime->transport_fraction);
        compute.destroyBuffer(runtime->contact_pairs);
        compute.destroyBuffer(runtime->wet_response);
        compute.destroyBuffer(runtime->dry_volume);
        for (const auto& lane : runtime->gradient) {
            for (const auto handle : lane) {
                compute.destroyBuffer(handle);
            }
        }
    }
