import mmap
import sys
import unittest

from tools.perception_bridge.video_card import (
    FRAME_MAP, HEADER, MAGIC, MAX_SIDE, VideoCardPublisher,
)


@unittest.skipUnless(sys.platform == "win32", "Player video mapping is Windows-only")
class VideoCardTransportTests(unittest.TestCase):
    def test_depth_contract_and_blank_on_close(self):
        import numpy as np
        publisher = VideoCardPublisher(synthetic_depth=0.75)
        self.addCleanup(publisher.close)
        with mmap.mmap(-1, HEADER.size + MAX_SIDE * MAX_SIDE * 8,
                       r"Local\RayTrophiDepthCard.v1", access=mmap.ACCESS_READ) as reader:
            pixels = np.array([[[10, 20, 30, 255], [1, 2, 3, 0]]], dtype=np.uint8)
            publisher.publish((pixels, 1.0, 2.0, 1.7))
            self.assertEqual(HEADER.unpack_from(reader)[-3:], (2, 1, 0x31445452))
            samples = np.frombuffer(reader[HEADER.size:HEADER.size + 16],
                                   dtype=[("rgba", "<u4"), ("depth", "<f4")])
            self.assertEqual(samples["rgba"].tolist(), [0xff1e140a, 0x00030201])
            self.assertEqual(samples["depth"].tolist(), [0.75, 0.75])
            publisher.publish((pixels, 1.0, 2.0, 1.7, np.array([[-2.0, 2.0]])))
            samples = np.frombuffer(reader[HEADER.size:HEADER.size + 16],
                                   dtype=[("rgba", "<u4"), ("depth", "<f4")])
            self.assertEqual(samples["depth"].tolist(), [-2.0, 2.0])
            serial = HEADER.unpack_from(reader)[0]
            for invalid in (np.array([[np.nan, 0]]), np.array([[21, 0]]), np.zeros((2, 2))):
                with self.assertRaises(ValueError):
                    publisher.publish((pixels, 1.0, 2.0, 1.7, invalid))
                self.assertEqual(HEADER.unpack_from(reader)[0], serial)
            publisher.close()
            self.assertEqual(reader[HEADER.size:HEADER.size + 16], bytes(16))

    def test_rgba_contract_exclusive_writer_and_blank_on_close(self):
        class Pixels:
            shape = (2, 3, 4)

            def tobytes(self):
                return bytes(range(24))

        publisher = VideoCardPublisher()
        self.addCleanup(publisher.close)
        with mmap.mmap(-1, HEADER.size + MAX_SIDE * MAX_SIDE * 4,
                       FRAME_MAP, access=mmap.ACCESS_READ) as reader:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                VideoCardPublisher()
            self.assertTrue(publisher.publish((Pixels(), 1.25, 2.5, 1.75)))
            serial, received, x, z, height, w, h, magic = HEADER.unpack_from(reader)
            self.assertGreater(received, 0)
            self.assertEqual((x, z, height, w, h, magic), (1.25, 2.5, 1.75, 3, 2, MAGIC))
            self.assertEqual(reader[HEADER.size:HEADER.size + 24], bytes(range(24)))
            publisher.close()
            blank = HEADER.unpack_from(reader)
            self.assertNotEqual(blank[0], serial)
            self.assertEqual(blank[-3:], (0, 0, MAGIC))
            self.assertEqual(reader[HEADER.size:HEADER.size + 24], bytes(24))
