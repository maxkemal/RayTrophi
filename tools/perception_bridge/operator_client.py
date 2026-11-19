"""External Python facade for the same service used by the Player operator panel.

Use outside the embedded script workspace so Player's main thread keeps running.
"""
try:
    from .transport import connect, PIPE
except ImportError:
    from transport import connect, PIPE


class PlayerOperator:
    def __init__(self, pipe=PIPE):
        self.pipe = pipe
        self.ipc = None

    def __enter__(self):
        self.ipc = connect(self.pipe)
        return self

    def __exit__(self, *_):
        if self.ipc is not None:
            self.ipc.close()
            self.ipc = None

    def command(self, operation="status", config=None):
        if self.ipc is None:
            raise RuntimeError("Use PlayerOperator as a context manager")
        return self.ipc.call("player.operator", operation=operation,
                             config={} if config is None else config)

    def status(self):
        return self.command()

    def add_folder(self, path):
        return self.command("library.add", {"path": str(path)})

    def open_project(self, path):
        return self.command("library.open", {"path": str(path)})

    def configure_camera(self, **settings):
        return self.command("camera.configure", settings)
