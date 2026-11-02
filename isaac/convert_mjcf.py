"""Convert the Kiko-Bot MJCF (models/robot.xml) into a USD asset for Isaac Sim.

Run once, with the Python that has Isaac Sim installed, and again whenever
robot.xml changes:

    python isaac/convert_mjcf.py                 # -> models/usd/kikobot.usd

The arm is imported fixed to the world (its base plate is bolted to the desk).
The joint drives are not taken from the import: kikobot_isaac_server.py
authors them from the MJCF's servo model each time it loads the arm.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("KIKOBOT_MODELS_DIR", REPO / "models"))


def sanitized_copy(robot_xml: Path) -> Path:
    """onshape-to-robot named the model and its default class after a Windows
    folder (`D:\\Ashoka University\\...`). Backslashes and colons in a name
    become invalid USD prim names, so import a copy with a plain name. The
    copy sits next to the original so `meshdir="assets"` still resolves."""
    text = robot_xml.read_text()
    text = re.sub(r'"[A-Za-z]:\\[^"]*"', '"kikobot"', text)
    fd, tmp = tempfile.mkstemp(prefix=".kikobot_isaac_", suffix=".xml", dir=robot_xml.parent)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return Path(tmp)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mjcf", default=str(MODELS / "robot.xml"))
    p.add_argument("--out", default=str(MODELS / "usd"), help="Output folder.")
    p.add_argument("--merge-mesh", action="store_true", help="Merge each link's meshes (faster render).")
    args = p.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp  # Must come before any other isaacsim import.

    app = SimulationApp({"headless": True})
    try:
        import omni.kit.app

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "isaacsim.asset.importer.mjcf", True
        )
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

        src = sanitized_copy(Path(args.mjcf))
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as work:
            try:
                cfg = MJCFImporterConfig(
                    mjcf_path=str(src),
                    usd_path=work,
                    fix_base=True,
                    import_scene=False,
                    merge_mesh=args.merge_mesh,
                )
                usd = Path(MJCFImporter(cfg).import_mjcf())
            finally:
                src.unlink(missing_ok=True)
            # The importer writes a main file plus layers/payloads beside it;
            # keep them together and give the main file a stable name.
            root = usd.parent
            for item in root.iterdir():
                dest = out / item.name
                if dest.exists():
                    shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                shutil.move(str(item), dest)
            main_usd = out / usd.name
            if main_usd.name != "kikobot.usd":
                stable = out / "kikobot.usd"
                stable.write_text(
                    "#usda 1.0\n(\n    defaultPrim = \"kikobot\"\n)\n\n"
                    f'def Xform "kikobot" (\n    prepend references = @./{main_usd.name}@\n)\n{{\n}}\n'
                )
        print(f"Wrote {out / 'kikobot.usd'} (imported as {main_usd.name}).")
    finally:
        app.close()


if __name__ == "__main__":
    main()
