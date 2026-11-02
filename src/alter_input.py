
class TrojanImage:
    def __init__(self,
        init_cap_pos0,
        init_cap_pos1,
        init_cfg_pos0,
        init_trigger_op,
        init_trigger_ref
    ): 
        self.curr_pos0 = init_cap_pos0
        self.curr_pos1 = init_cap_pos1

        self.curr_cfg_pos = init_cfg_pos0

        self.curr_trigger_op  = init_trigger_op
        self.curr_trigger_ref = init_trigger_ref

        self.position_codes = {
            "DRIVE"     : { "str" : "0001", "int" : 0b0001 },
            "OFFSET"    : { "str" : "0010", "int" : 0b0010 },
            "TARGET"    : { "str" : "0011", "int" : 0b0011 },
            "REFERENCE" : { "str" : "0100", "int" : 0b0100 },
            "SELF POS"  : { "str" : "0101", "int" : 0b0101 },
            "TRIG POS"  : { "str" : "0110", "int" : 0b0110 },
            "READ"      : { "str" : "0111", "int" : 0b0111 },
            "WRITE"     : { "str" : "1000", "int" : 0b1000 },

            "OTHERWISE" : { "str" : "0000", "int" : 0b0000 }
        }

        self.trigger_codes = {
            "XNOR" : { "str" : "0", "int" : 0b0},
            "XOR"  : { "str" : "1", "int" : 0b1}
        }

        self.payload_codes = {
            "OFFSET"  : { "str" : "00", "int" : 0b00 },
            "DRIVE"   : { "str" : "01", "int" : 0b01 },
            "READ_W"  : { "str" : "10", "int" : 0b10 },
            "WRITE_W" : { "str" : "11", "int" : 0b11 },
        }

        self.cfg_update_code = {
            "NO"  : { "str" : "0", "int" : 0b0},
            "YES" : { "str" : "1", "int" : 0b1}
        }

        # ABBA + 1 -- note it will have the following hex: 15775
        self.config_key = {"str" : "10101011101110101", "int" : 0b10101011101110101 }

    def update_config(self,

        # DRIVE, OFFSET, TARGET, REFERENCE, SELF POS, TRIG POS, READ, WRITE
        type_update_pos0 = None,
        type_update_pos1 = None,

        what_pos0_is = None,
        what_pos1_is = None,

        type_trigger = None, # XNOR, XOR
        type_payload = None, # OFFSET, DRIVE, READ_W, WRITE_W

        which_weight = None, # which weight to read from or write to

        change_config_location = None, # YES, NO (uses config pos + 1)
        new_config_pos = None # place the new config should be seen at
    ):
        cfg_word = timage.make_cfg_word(
            type_update_pos0       = type_update_pos0,
            type_update_pos1       = type_update_pos1,
            type_payload           = type_payload,
            type_trigger           = type_trigger,
            change_config_location = change_config_location
        ) 
        
        # write pos0, pos1, cfg, and cfg self update

        pass

    def make_cfg_word(self,
        type_update_pos0       = None,
        type_update_pos1       = None,
        type_payload           = None,
        type_trigger           = None,
        change_config_location = None
    ):
        # build a configure word (update cfg self update word if needed)
        # add pos decoding
        cfg_word = ""
        for pos in [type_update_pos0, type_update_pos1]:
            cfg_word += self.position_codes[pos]["str"]
        cfg_word += "000"
        cfg_word += self.payload_codes[type_payload]["str"]
        cfg_word += self.trigger_codes[type_trigger]["str"]
        cfg_word += self.cfg_update_code[change_config_location]["str"]
        cfg_word += self.config_key["str"]

        return cfg_word

    def alter_image(self,
        image, # normalized tensor

        trojan_state, # to configure trigger

        # if the trojan needs to be updated
        update_trojan = False,
    ):
        pass


if __name__ == "__main__":
    timage = TrojanImage(
        init_cap_pos0    = None,
        init_cap_pos1    = None,
        init_cfg_pos0    = None,
        init_trigger_op  = None,
        init_trigger_ref = None
    )

    cfg = timage.make_cfg_word(
        type_update_pos0       = "OFFSET",
        type_update_pos1       = "DRIVE",
        type_payload           = "DRIVE",
        type_trigger           = "XNOR",
        change_config_location = "NO"
    )

    print(cfg)

    print(len(cfg))
