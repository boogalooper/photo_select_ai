from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

from app.core.models import PhotoFile
from app.xmp.writer import XMP_NS, XmpWriter, _find_standard_xmp_payload, _parse_xmp_payload


CFG = {
    "xmp": {
        "scheme":"bridge",
        "bridge_red":"Select",
        "bridge_yellow":"Second",
        "lightroom_red":"Red",
        "lightroom_yellow":"Yellow",
        "custom_red":"R",
        "custom_yellow":"Y",
        "jpeg_embedded": True,
        "update_existing_jpeg_sidecar": True,
    }
}


class XmpTests(unittest.TestCase):
    def test_label_write_sidecar_raw(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.CR3"
            photo_path.write_bytes(b"x")
            p = PhotoFile(photo_path, datetime.now(), 1, ".cr3")
            xmp = XmpWriter(CFG).write(p, "red")
            root = ET.parse(xmp).getroot()
            desc = next(root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertEqual(desc.attrib[f"{{{XMP_NS}}}Label"], "Select")

    def test_preserves_existing_attribute(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.CR3"
            photo_path.write_bytes(b"x")
            xmp_path = Path(td) / "A.xmp"
            xmp_path.write_text(
                '<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"><rdf:RDF><rdf:Description rdf:about="" crs:Exposure2012="0.55"/></rdf:RDF></x:xmpmeta>',
                encoding="utf-8",
            )
            p = PhotoFile(photo_path, datetime.now(), 1, ".cr3")
            XmpWriter(CFG).write(p, "red")
            root = ET.parse(xmp_path).getroot()
            desc = next(root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertEqual(desc.attrib["{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012"], "0.55")
            self.assertEqual(desc.attrib[f"{{{XMP_NS}}}Label"], "Select")

    def test_jpeg_label_is_embedded_and_image_remains_readable(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.jpg"
            Image.new("RGB", (64, 48), (120, 80, 30)).save(photo_path, "JPEG")
            p = PhotoFile(photo_path, datetime.now(), 1, ".jpg")
            result = XmpWriter(CFG).write(p, "red")
            self.assertEqual(result, photo_path)
            self.assertFalse((Path(td) / "A.xmp").exists())

            payload = _find_standard_xmp_payload(photo_path.read_bytes())
            self.assertIsNotNone(payload)
            tree = _parse_xmp_payload(payload)
            desc = next(tree.getroot().iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertEqual(desc.attrib[f"{{{XMP_NS}}}Label"], "Select")

            with Image.open(photo_path) as im:
                self.assertEqual(im.size, (64, 48))


    def test_jpeg_yellow_role_writes_yellow_label(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "Y.jpg"
            Image.new("RGB", (32, 32), (30, 40, 50)).save(photo_path, "JPEG")
            p = PhotoFile(photo_path, datetime.now(), 1, ".jpg")
            XmpWriter(CFG).write(p, "yellow")
            payload = _find_standard_xmp_payload(photo_path.read_bytes())
            self.assertIsNotNone(payload)
            tree = _parse_xmp_payload(payload)
            desc = next(tree.getroot().iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertEqual(desc.attrib[f"{{{XMP_NS}}}Label"], "Second")

    def test_jpeg_existing_sidecar_is_updated_but_new_one_is_not_created(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.jpg"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(photo_path, "JPEG")
            sidecar = Path(td) / "A.xmp"
            sidecar.write_text(
                '<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:RDF><rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>',
                encoding="utf-8",
            )
            p = PhotoFile(photo_path, datetime.now(), 1, ".jpg")
            XmpWriter(CFG).write(p, "red")
            root = ET.parse(sidecar).getroot()
            desc = next(root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertEqual(desc.attrib[f"{{{XMP_NS}}}Label"], "Select")

    def test_clear_red_label_preserves_other_sidecar_metadata(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.CR3"
            photo_path.write_bytes(b"x")
            p = PhotoFile(photo_path, datetime.now(), 1, ".cr3")
            writer = XmpWriter(CFG)
            writer.write(p, "red")
            xmp_path = photo_path.with_suffix(".xmp")
            tree = ET.parse(xmp_path)
            desc = next(tree.getroot().iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            desc.set("{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012", "0.25")
            tree.write(xmp_path, encoding="utf-8", xml_declaration=True)
            self.assertTrue(writer.clear_red_label(p))
            root = ET.parse(xmp_path).getroot()
            desc = next(root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertNotIn(f"{{{XMP_NS}}}Label", desc.attrib)
            self.assertEqual(desc.attrib["{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012"], "0.25")

    def test_clear_red_label_from_embedded_jpeg(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.jpg"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(photo_path, "JPEG")
            p = PhotoFile(photo_path, datetime.now(), 1, ".jpg")
            writer = XmpWriter(CFG)
            writer.write(p, "red")
            self.assertTrue(writer.clear_red_label(p))
            payload = _find_standard_xmp_payload(photo_path.read_bytes())
            self.assertIsNotNone(payload)
            tree = _parse_xmp_payload(payload)
            desc = next(tree.getroot().iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertNotIn(f"{{{XMP_NS}}}Label", desc.attrib)
            with Image.open(photo_path) as im:
                self.assertEqual(im.size, (32, 32))


if __name__ == "__main__":
    unittest.main()
