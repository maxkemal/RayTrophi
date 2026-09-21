"""Run in the rebuilt app: test_material_coverage(existing_material_name).

Uses one existing material and restores its authored setting, even on failure.
"""
import rt

def test_material_coverage(material_name):
    old = rt.materials.get_param(material_name, 'alpha_cutout')
    transmission = rt.materials.get_param(material_name, 'transmission')
    try:
        for value in (1, 0, 1):
            rt.materials.set_param(material_name, 'alpha_cutout', value)
            assert rt.materials.get_param(material_name, 'alpha_cutout') == value
            assert rt.materials.get_param(material_name, 'transmission') == transmission
        for invalid in (-1, 0.5, 2, float('nan'), float('inf')):
            try:
                rt.materials.set_param(material_name, 'alpha_cutout', invalid)
            except Exception:
                pass
            else:
                raise AssertionError(f'Accepted invalid alpha_cutout={invalid}')
            assert rt.materials.get_param(material_name, 'alpha_cutout') == 1
    finally:
        rt.materials.set_param(material_name, 'alpha_cutout', old)
    assert rt.materials.get_param(material_name, 'alpha_cutout') == old
