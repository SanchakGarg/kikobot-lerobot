"""Scripted, verified demonstrations on the desk, and their conversion for OpenVLA.

    kikobot-datagen record  --task pen_to_pen_cup --episodes 200 --out datasets/kikobot_desk
    kikobot-datagen convert --out datasets/kikobot_desk --task pen_to_pen_cup

`tasks.py` lists the tasks (one kind of item into its station), `expert.py`
is the scripted oracle that does them with privileged sim state, `checks.py`
decides whether an episode was perfect, `record.py` keeps only the perfect
ones, and `rlds/` turns them into an RLDS (TFDS) dataset. See
docs/datagen-openvla.md.
"""
