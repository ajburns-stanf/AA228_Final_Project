# === Initialize STK ===
import os


def setup():
    # Need to ensure that the pystk2 DLLs are available since they are not bundled automatically on Windows
    # with the wheel, so we borrow them from the SuperTuxKart instal.
    if os.name == "nt":
        os.add_dll_directory(r"C:\Program Files\SuperTuxKart 1.5")
