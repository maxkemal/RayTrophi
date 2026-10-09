"""Independent CPU oracle for the device page-remap/retirement transaction.

Does not execute C++/GLSL. A later user build and GPU probe must verify both.
"""

import copy
import random


def transaction(old, new_keys, backgrounds):
    old_keys, old_fields = old
    old_slots = {key: slot for slot, key in enumerate(old_keys)}
    new_map = {key: slot + 1 for slot, key in enumerate(new_keys)}
    candidate = []
    invalid = False
    for pages, background in zip(old_fields, backgrounds):
        for key, page in zip(old_keys, pages):
            if key not in new_map and any(value != background for value in page):
                invalid = True
        next_pages = []
        stride = len(pages[0])
        for key in new_keys:
            if key in old_slots:
                next_pages.append(pages[old_slots[key]][:])
            else:
                next_pages.append([background] * stride)
        candidate.append(next_pages)
    if invalid:
        raise ValueError("live physical channel would be discarded")
    return new_keys[:], candidate


def main():
    randomizer = random.Random(8)
    keys = [3, 7, 23]
    backgrounds = [0.0, 293.15, 1.0, 0.0]
    strides = [512, 512, 576, 576]
    fields = [[[background] * stride for _ in keys]
              for background, stride in zip(backgrounds, strides)]
    fields[0][0][1] = 0.013
    fields[1][1][12] = 350.0
    fields[2][1][321] = 0.25
    fields[3][0][255] = -4.0
    original = (keys, fields)
    frozen = copy.deepcopy(original)
    for _ in range(20):
        new_keys = keys[:] + [31]
        randomizer.shuffle(new_keys)
        next_state = transaction(original, new_keys, backgrounds)
        for old_slot, key in enumerate(keys):
            new_slot = new_keys.index(key)
            for channel in range(len(fields)):
                assert next_state[1][channel][new_slot] == fields[channel][old_slot]
        for channel, background in enumerate(backgrounds):
            assert next_state[1][channel][new_keys.index(31)] == [background] * strides[channel]
        assert original == frozen
    for retired in (3, 7):
        try:
            transaction(original, [key for key in keys if key != retired], backgrounds)
        except ValueError:
            pass
        else:
            raise AssertionError("A later live channel was discarded")
        assert original == frozen
    # A fully background page may retire even when other channels are live.
    next_state = transaction(original, [7, 3], backgrounds)
    assert next_state[0] == [7, 3]
    # Clearing density alone is never permission to discard heat/velocity.
    no_density = copy.deepcopy(original)
    no_density[1][0] = [[0.0] * 512 for _ in keys]
    try:
        transaction(no_density, [3, 23], backgrounds)
    except ValueError:
        pass
    else:
        raise AssertionError("Smoke-only retirement lost invisible thermal/pressure support")
    print("PASS CPU all-channel page remap, slot permutations, backgrounds and atomic rejection")


if __name__ == "__main__":
    main()
