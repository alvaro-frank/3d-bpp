from environment.packing_env import PackingEnv
from environment.box import Box
from heuristics.heuristic import heuristic_blb_packing

def evaluate_heuristic_on_episode(episode_boxes, env_seed=None, generate_gif=False, gif_name="packing_heuristic.gif"):
    """
    Evaluate the heuristic baseline on ONE test episode.

    - Recreates the environment to ensure fairness.
    - Uses a bottom-left-back heuristic with optional rotations.
    - Optionally generates a GIF of the packing process.

    Parameters:
    - episode_boxes (list[dict]): list of boxes with dimensions (w, h, d)
    - env_seed (int, optional): seed for reproducibility
    - generate_gif (bool): whether to record GIF
    - gif_name (str): filename for output GIF

    Returns:
    - float: percentage of bin volume used
    """
    env = PackingEnv(bin_size=(10, 10, 10), max_boxes=len(episode_boxes))
    env.reset(seed=env_seed, with_boxes=episode_boxes)

    # Convert dicts -> Box objects for heuristic
    boxes_for_heur = [Box(b["w"], b["h"], b["d"], id=i) for i, b in enumerate(episode_boxes)]

    placed_boxes, _bin = heuristic_blb_packing(
        bin_size=env.bin_size,
        boxes=boxes_for_heur,
        try_rotations=True,
        generate_gif=generate_gif,
        gif_name=gif_name,
    )

    volume_used = sum(box.get_volume() for box in placed_boxes)
    pct_volume_used = (volume_used / env.bin.bin_volume()) * 100.0
    return pct_volume_used
