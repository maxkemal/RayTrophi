"""Embedded production API check. Imports must already exist; adds one undoable clip."""


def run(source_character, source_clip, target_character, node_map=None, mode="same_rig", translation_scale=1.0):
    import rt
    from rig_mapping_preview_contract import check_sample
    node_map = {} if node_map is None else node_map
    check_sample(lambda method, values: getattr(rt.anim, method.split(".")[1])(**values),
                 dict(source_character=source_character, source_clip=source_clip,
                      target_character=target_character, node_map=node_map,
                      mode=mode, translation_scale=translation_scale))
    before_source = rt.anim.source_channels(source_clip)
    before_target = rt.anim.clips(target_character)
    preview = rt.anim.preview_clip_binding(source_character, source_clip, target_character, node_map=node_map, mode=mode, translation_scale=translation_scale)
    assert preview["ready"], preview
    assert preview["mode"] == mode
    assert preview["translation_scale"] == translation_scale
    bound = rt.anim.bind_clip(source_character, source_clip, target_character, node_map=node_map, mode=mode, translation_scale=translation_scale)
    assert bound["ready"] and bound["output_clip"] != source_clip
    assert bound["mode"] == mode
    assert bound["matches"] == preview["matches"]
    after_target = rt.anim.clips(target_character)
    assert len(after_target) == len(before_target) + 1
    assert any(clip["name"] == bound["output_clip"] for clip in after_target)
    assert rt.anim.source_channels(source_clip) == before_source
    output_channels = rt.anim.source_channels(bound["output_clip"])
    mapped = {match["target"] for match in bound["matches"]}
    assert {channel["node_name"] for channel in output_channels} <= mapped
    print("PASS: target clip registration, source preservation, channel remap", bound["output_clip"])
    return bound
