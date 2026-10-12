from example import make_config_pixel
import torch

# it works!
if __name__ == "__main__":
    cfg_pixel = make_config_pixel(
        pos0 = "none",
        pos1 = "none",

        payload_type   = "drive",
        operation_type = "XNOR",
        self_program   = "NO",

        newx = 0,
        newy = 0
    )

    print(cfg_pixel)
    print(f"{cfg_pixel[0]:032b}")
    print(f"abcdefghijklmnopqrstuvwxyzabcdef")
