#include "../RayTrophiStudio/source/include/Backend/VulkanMaterialTextureDiagnostics.h"
#include "../RayTrophiStudio/source/include/Backend/VulkanMaterialTextureFramePublication.h"
#include "../RayTrophiStudio/source/include/Backend/VulkanMaterialTextureResidency.h"
#include "../RayTrophiStudio/source/include/Backend/VulkanMaterialTextureUploadQueue.h"

#include <cassert>

using namespace Backend::MaterialTexturePages;

static VKAPI_ATTR VkResult VKAPI_CALL queryImage(
    VkPhysicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling,
    VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties* out) {
    assert(type == VK_IMAGE_TYPE_2D && tiling == VK_IMAGE_TILING_OPTIMAL);
    assert(usage == (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    assert(flags == 0);
    if (format != VK_FORMAT_R8G8B8A8_SRGB) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    *out = {{2048, 1024, 1}, 12, 4096, VK_SAMPLE_COUNT_1_BIT, 2ull << 30};
    return VK_SUCCESS;
}

struct TestSnapshot {
    VkDescriptorSet set = VK_NULL_HANDLE;

    VkDescriptorSet descriptorSet() const {
        return set;
    }
};

int main() {
    const auto dummySampler = reinterpret_cast<VkSampler>(uintptr_t{1});
    const auto pageSampler = reinterpret_cast<VkSampler>(uintptr_t{2});
    ArrayImage dummy;
    dummy.view = reinterpret_cast<VkImageView>(uintptr_t{1});
    dummy.key = {0, 1, 1, 1};
    dummy.layers = 1;
    ArrayImage image;
    image.view = reinterpret_cast<VkImageView>(uintptr_t{2});
    image.layers = 2;
    PublishedPages pages{};
    pages[0] = {&image, pageSampler, {LayerState::Resident, LayerState::Unloaded}};
    const std::vector<uint32_t> references{packRef(0, 0), kNoTexture};
    DescriptorImages descriptors{};
    assert(prepareDescriptorImages(pages, references, dummy, dummySampler, descriptors));
    assert(descriptors[0].imageView == image.view && descriptors[0].sampler == pageSampler);
    for (std::size_t i = 1; i < descriptors.size(); ++i) {
        assert(descriptors[i].imageView == dummy.view && descriptors[i].sampler == dummySampler);
        assert(descriptors[i].imageLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    const auto write = descriptorWrite(VK_NULL_HANDLE, 7, descriptors);
    assert(write.descriptorCount == 32 && write.dstArrayElement == 0 && write.dstBinding == 7);
    assert(write.pImageInfo == descriptors.data());
    const auto unloaded = prepareDescriptorImages(
        pages, {packRef(0, 0), packRef(0, 1)}, dummy, dummySampler, descriptors);
    assert(unloaded.reason == DescriptorRejection::LayerUnloaded && unloaded.referenceIndex == 1u);
    const auto outside = prepareDescriptorImages(
        pages, {packRef(0, 2)}, dummy, dummySampler, descriptors);
    assert(outside.reason == DescriptorRejection::LayerOutOfRange && outside.layer == 2u);
    const auto missing = prepareDescriptorImages(
        pages, {packRef(1, 0)}, dummy, dummySampler, descriptors);
    assert(missing.reason == DescriptorRejection::MissingPage && missing.page == 1u);
    pages[0].layers[1] = LayerState::Retired;
    const auto retired = prepareDescriptorImages(
        pages, {packRef(0, 1)}, dummy, dummySampler, descriptors);
    assert(retired.reason == DescriptorRejection::LayerRetired && retired.reference == 1u);
    DescriptorPublicationDiagnostics publications;
    assert(publications.snapshot().at("last_descriptor_rejection").is_null());
    std::string log;
    publications.recordRejection(retired, [&](const std::string& message) { log = message; });
    const auto publicationStatus = publications.snapshot();
    assert(publicationStatus.at("descriptor_publications_rejected") == 1);
    assert(publicationStatus.at("last_descriptor_rejection").at("reason") == "layer_retired");
    assert(publicationStatus.at("last_descriptor_rejection") == nlohmann::json::parse(log));
    assert(descriptors[0].imageView == image.view);

    FramePublication<TestSnapshot> publication;
    PageResidency residency(0, 2);
    const auto initial = residency.reserve();
    assert(initial.ref == packRef(0, 0));
    assert(residency.markUploadSubmitted(initial, 8));
    UploadQueueStatus uploads{2, 7, 3};
    assert(uploadQueueDiagnostics(uploads).at("pending_uploads") == 2);
    assert(uploadQueueDiagnostics(uploads).at("completed_upload_serial") == 7);
    assert(uploadQueueDiagnostics(uploads).at("upload_not_ready_rejections") == 3);
    unsigned allocated = 0;
    int prepared = 0;
    VkDescriptorSet alias = VK_NULL_HANDLE;
    bool prepareSucceeds = true;
    const auto allocate = [&] {
        ++allocated;
        const auto set = reinterpret_cast<VkDescriptorSet>(static_cast<uintptr_t>(allocated));
        return std::make_shared<TestSnapshot>(TestSnapshot{alias ? alias : set});
    };
    const auto prepare = [&](TestSnapshot&, const DescriptorImages&) {
        ++prepared;
        return prepareSucceeds;
    };
    const auto publish = [&](uint64_t revision, uint64_t wanted, LayerReservation replacement,
                             LayerReservation previous,
                             const std::vector<uint32_t>& refs) {
        return publication.publish(revision, wanted, residency, replacement, previous, uploads,
                                   pages, refs, dummy, dummySampler, allocate, prepare,
                                   [](const std::string&) {});
    };
    assert(publish(1, 1, initial, {}, references) == FramePublicationResult::UploadPending);
    assert(publish(1, 2, initial, {}, references) == FramePublicationResult::StaleRevision);
    assert(allocated == 0);
    uploads.completedUploadSerial = 8;
    assert(publish(1, 1, initial, {}, references) == FramePublicationResult::Published);
    const std::weak_ptr<const TestSnapshot> first = publication.current();
    publication.retainReader({10}, publication.current());
    const auto replacement = residency.reserve();
    assert(replacement.ref == packRef(0, 1));
    assert(residency.markUploadSubmitted(replacement, 9));
    uploads.completedUploadSerial = 100;
    assert(publish(2, 2, replacement, initial, references) ==
           FramePublicationResult::InvalidDescriptors);
    assert(publication.diagnostics().rejectedPublications == 1);
    assert(publication.current() == first.lock() && allocated == 1 && prepared == 1);
    alias = publication.current()->descriptorSet();
    assert(publish(2, 2, replacement, initial, {replacement.ref}) ==
           FramePublicationResult::ResourcePreparationFailed);
    assert(prepared == 1);
    alias = VK_NULL_HANDLE;
    prepareSucceeds = false;
    assert(publish(2, 2, replacement, initial, {replacement.ref}) ==
           FramePublicationResult::ResourcePreparationFailed);
    assert(publication.current() == first.lock());
    assert(residency.describe()[0] == LayerState::Resident);
    assert(residency.describe()[1] == LayerState::Unloaded);
    prepareSucceeds = true;
    publication.retainReader({13}, publication.current());
    assert(publish(2, 2, replacement, initial, {replacement.ref}) ==
           FramePublicationResult::Published);
    assert(publication.current()->descriptorSet() != first.lock()->descriptorSet());
    assert(residency.describe()[0] == LayerState::Retired);
    assert(residency.describe()[1] == LayerState::Resident);
    publication.reclaimReaders({9});
    assert(!first.expired());
    publication.reclaimReaders({12});
    residency.reclaim(100, 12);
    assert(!first.expired() && residency.reserve().ref == kNoTexture);
    publication.reclaimReaders({13});
    assert(first.expired());
    residency.reclaim(100, 13);
    const auto retry = residency.reserve();
    assert(retry.ref == initial.ref && retry.version != initial.version);
    assert(!residency.markUploadSubmitted(initial, 101));
    assert(residency.markUploadSubmitted(retry, 102));
    assert(residency.discard(retry));
    residency.reclaim(101, 100);
    assert(residency.reserve().ref == kNoTexture);
    residency.reclaim(102, 100);
    assert(residency.reserve().ref == initial.ref);

    PageResidency nextPage(1, 2);
    const auto migrated = nextPage.reserve();
    assert(nextPage.markUploadSubmitted(migrated, 103));
    uploads.completedUploadSerial = 103;
    PublishedPages migrationPages = pages;
    migrationPages[1] = pages[0];
    const auto migrate = [&](const std::vector<uint32_t>& refs) {
        return publication.publish(3, 3, nextPage, migrated, replacement, uploads,
                                   migrationPages, refs, dummy, dummySampler, allocate,
                                   prepare, [](const std::string&) {}, &residency);
    };
    publication.retainReader({14}, publication.current());
    assert(migrate({replacement.ref, migrated.ref}) ==
           FramePublicationResult::InvalidDescriptors);
    assert(residency.describe()[1] == LayerState::Resident);
    assert(nextPage.describe()[0] == LayerState::Unloaded);
    assert(migrate({migrated.ref}) == FramePublicationResult::Published);
    residency.reclaim(103, 13);
    assert(residency.describe()[1] == LayerState::Retired);
    assert(nextPage.describe()[0] == LayerState::Resident);
    publication.reclaimReaders({14});
    residency.reclaim(103, 14);
    assert(residency.describe()[1] == LayerState::Unloaded);

    const auto layers = layerDiagnostics(pages, references);
    assert(layers.at("layers_used") == 1 && layers.at("layers_capacity") == 2);
    assert(layers.at("layers_valid") == true);
    assert(layerDiagnostics(pages, {packRef(0, 1)}).at("layers_valid") == false);
    pages[0].layers = {LayerState::Resident, LayerState::Resident, LayerState::Resident};
    assert(layerDiagnostics(pages, references).at("layers_valid") == false);
    assert(!prepareDescriptorImages(pages, references, dummy, dummySampler, descriptors));

    const auto queryLayers = [](const Key&) { return 4096u; };
    const auto payloadBytes = [](const Key& key) {
        uint64_t bytes = 0;
        for (uint32_t mip = 0; mip < key.mipLevels; ++mip) {
            const uint32_t width = (std::max)(1u, key.width >> mip);
            const uint32_t height = (std::max)(1u, key.height >> mip);
            bytes += 4ull * width * height;
        }
        return bytes;
    };

    const std::vector<Texture> omix(1025, {{1, 512, 512, 10}, 1});
    const Plan split = makePlan(omix, 1024, queryLayers, payloadBytes);
    assert(split.error == PlanError::None && split.pages.size() == 2);
    assert(split.pages[0].textures.size() == 1024);
    assert(split.refs.front() == 0 && split.refs.back() == packRef(1, 0));
    assert(split.mergedKeys.empty());

    VkPhysicalDeviceLimits deviceLimits{};
    deviceLimits.maxImageDimension2D = 4096;
    deviceLimits.maxImageArrayLayers = 1024;
    constexpr auto usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const ImageLimits imageLimits = queryImageLimits(
        VK_NULL_HANDLE, deviceLimits,
        {{VK_FORMAT_R8G8B8A8_SRGB, usage}, {VK_FORMAT_BC7_SRGB_BLOCK, usage}}, queryImage);
    assert(imageLimits.layersFor({0, 2048, 1024, 12}) == 1024);
    assert(imageLimits.layersFor({0, 2048, 1025, 12}) == 0);
    assert(imageLimits.layersFor({0, 128, 128, 9}) == 0);
    assert(imageLimits.layersFor({1, 128, 128, 8}) == 0);
    const Plan partial = makeVulkanPlan(
        {{{0, 512, 512, 10}, 1},
         {{0, 8192, 8192, 14}, 1},
         {{1, 128, 128, 8}, 1},
         {{0, 512, 512, 10}, 1}},
        imageLimits, payloadBytes);
    assert(partial.error == PlanError::None && partial.pages.size() == 1);
    assert(partial.refs[0] == packRef(0, 0) && partial.refs[3] == packRef(0, 1));
    assert(partial.refs[1] == kNoTexture && partial.refs[2] == kNoTexture);
    assert(partial.rejectedTextures.size() == 2);
    const auto diagnostics = imageDiagnostics(imageLimits, partial);
    const auto& rejectedQuery = diagnostics.at("image_format_queries").at(1);
    assert(rejectedQuery.at("format") == static_cast<int>(VK_FORMAT_BC7_SRGB_BLOCK));
    assert(rejectedQuery.at("usage_flags") == usage);
    assert(rejectedQuery.at("vk_result") == static_cast<int>(VK_ERROR_FORMAT_NOT_SUPPORTED));
    assert(rejectedQuery.at("properties").is_null());
    assert(diagnostics.at("warnings").at(0).at("texture_index") == 1);
    assert(diagnostics.at("warnings").at(1).at("profile") == 1);
    assert(diagnostics.at("rejected_textures").at(0).at("vk_result") == VK_SUCCESS);
    assert(diagnostics.at("rejected_textures").at(1).at("vk_result") ==
           static_cast<int>(VK_ERROR_FORMAT_NOT_SUPPORTED));
    const Plan actualLimits = makeVulkanPlan(
        std::vector<Texture>(1025, {{0, 512, 512, 10}, 1}), imageLimits, payloadBytes);
    assert(actualLimits.error == PlanError::None && actualLimits.pages.size() == 2);
    const Plan sizeLimited = makeVulkanPlan(
        std::vector<Texture>(400, {{0, 2048, 1024, 12}, 1}), imageLimits, payloadBytes);
    assert(sizeLimited.error == PlanError::None && sizeLimited.pages.size() == 3);

    const Plan formatLimited = makePlan(omix, 4096, [](const Key&) { return 512u; },
                                       payloadBytes);
    assert(formatLimited.error == PlanError::None && formatLimited.pages.size() == 3);

    std::vector<Texture> mixed;
    for (uint32_t i = 0; i < 33; ++i) {
        const uint32_t referenceCount = i == 0 ? 1u : 10u;
        mixed.push_back({{1, 300u + i, 400u + i, 9}, referenceCount});
    }
    const Plan overflow = makePlan(mixed, 1024, queryLayers, payloadBytes);
    assert(overflow.error == PlanError::None && overflow.pages.size() == 32);
    assert(overflow.mergedKeys.size() == 1);
    assert(overflow.mergedKeys[0].original == mixed[0].key);
    assert(overflow.mergedKeys[0].target == mixed[1].key);
    assert(overflow.mergedKeys[0].extraPayloadBytes > 0);

    for (std::size_t i = 0; i < mixed.size(); ++i) {
        const uint32_t ref = overflow.refs[i];
        assert(ref != kNoTexture);
        const Page& page = overflow.pages.at(ref >> 16);
        assert(page.textures.at(ref & 0xffffu) == i);
        assert(page.key.width >= mixed[i].key.width);
        assert(page.key.height >= mixed[i].key.height);
        assert(page.key.mipLevels >= mixed[i].key.mipLevels);
    }

    for (uint32_t i = 0; i < mixed.size(); ++i) {
        mixed[i].key.compatibility = i;
    }
    const Plan incompatible = makePlan(mixed, 1024, queryLayers, payloadBytes);
    assert(incompatible.error == PlanError::IncompatiblePageBudget);
    assert(incompatible.pages.empty() && incompatible.refs.empty());
}
