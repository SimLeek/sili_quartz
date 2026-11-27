

def test_last_pause_time_reads_the_last_pause_line_of_the_game_log():
    import importlib.util
    import time
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools" / "minecraft.py"
    spec = importlib.util.spec_from_file_location("mc_tools", path)
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    tail = ("[27Nov2026 22:10:00.001] [Server thread/INFO] [x]: Saving and pausing game...\n"
            "[27Nov2026 22:20:00.002] [Render thread/INFO] [y]: something else\n"
            "[27Nov2026 22:26:31.608] [Server thread/INFO] [x]: Saving and pausing game...\n"
            "[27Nov2026 22:26:31.609] [Server thread/INFO] [z]: Saving chunks for level\n")
    t = mc.last_pause_time(tail)
    assert t == time.mktime(time.strptime("27Nov2026 22:26:31", "%d%b%Y %H:%M:%S"))
    assert mc.last_pause_time("[27Nov2026 22:20:00.002] no pause here\n") is None
