"""External Player language acceptance after build. --switch changes/restores locale.

Does not open scenes, cameras or save a project. Run from a separate terminal.
"""
import argparse
from rt_ipc import RtIpc, RtIpcError, PLAYER_PIPE_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switch", action="store_true")
    args = parser.parse_args()
    ipc = RtIpc(PLAYER_PIPE_NAME)

    def command(operation, config=None):
        return ipc.call("player.operator", operation=operation,
                        config={} if config is None else config)

    before = command("language.status")
    assert {item["code"] for item in before["available"]} >= {"en", "tr"}
    for config in ({"locale": "../invalid"}, {"locale": True},
                   {"locale": "tr", "unexpected": 1}):
        try:
            command("language.set", config)
        except RtIpcError:
            pass
        else:
            raise AssertionError(f"Invalid locale accepted: {config}")
        assert command("language.status")["locale"] == before["locale"]
    print("PASS: locale registry and invalid-setting invariance")
    if args.switch:
        state = command("status")
        try:
            for locale in ("tr", "en", "tr"):
                assert command("language.set", {"locale": locale})["locale"] == locale
                current = command("status")
                assert current["preferences"]["locale"] == locale
                assert current["game_settings"]["values"] == state["game_settings"]["values"]
                assert current["project"] == state["project"]
            print("PASS: EN/TR switching persists preference and preserves gameplay settings")
        finally:
            command("language.set", {"locale": before["locale"]})
    ipc.close()


if __name__ == "__main__":
    main()
