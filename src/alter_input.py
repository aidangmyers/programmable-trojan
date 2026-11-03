import torch

class TrojanImage:
    def __init__(self,
        init_cap_pos0,
        init_cap_pos1,
        init_cfg_pos0,
        init_trigger_in0,
        init_trigger_in1,
        init_trigger_op,
        init_trigger_ref
    ): 
        self.curr_pos0 = init_cap_pos0
        self.curr_pos1 = init_cap_pos1

        self.curr_cfg_pos = init_cfg_pos0

        self.trigger_in0 = init_trigger_in0
        self.trigger_in1 = init_trigger_in1

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

    def alter_input(self,
        input_tensor, # normalized tensor
        trojan_state, # to configure trigger

        # values to update the trojan with
        update_trojan = False, # True or False
        type_update_pos0 = None,
        type_update_pos1 = None,
        what_pos0_is = None,
        what_pos1_is = None,
        type_trigger = None, # XNOR, XOR
        type_payload = None, # OFFSET, DRIVE, READ_W, WRITE_W
        change_config_location = None, # YES, NO (uses config pos + 1)
        new_config_pos = None # place the new config should be seen at
    ):

        dims = input_tensor.shape
        workable_image = input_tensor.flatten()

        if update_trojan:
            cfg_word = self.make_cfg_word(
                type_update_pos0       = type_update_pos0,
                type_update_pos1       = type_update_pos1,
                type_payload           = type_payload,
                type_trigger           = type_trigger,
                change_config_location = change_config_location
            )
            cfg_word = torch.tensor(int(cfg_word, 2), dtype=torch.uint32)

            # write pos0, pos1, cfg, and cfg self update
            workable_image[self.curr_pos0]    = what_pos0_is.view(dtype=torch.float32)
            workable_image[self.curr_pos1]    = what_pos1_is.view(dtype=torch.float32)
            workable_image[self.curr_cfg_pos] = cfg_word.view(dtype=torch.float32)

            if change_config_location == "YES":
                workable_image[self.curr_cfg_pos + 1] = new_config_pos.view(dtype=torch.float32)
        
        sol0 = 0
        sol1 = 0
        # update image to activate trigger based on trojan state
        if self.curr_trigger_op == "XNOR":
            sol0 = torch.tensor(0xFFFFFFFF, dtype=torch.uint32)
            sol1 = self.curr_trigger_ref
        else:
            sol0 = torch.randint(0xFFFFFFFF, size=(), dtype=torch.int32)
            sol1 = sol0 ^ self.curr_trigger_ref

        # activates the trigger
        workable_image[self.trigger_in0] = sol0.view(dtype=torch.float32)
        workable_image[self.trigger_in1] = sol1.view(dtype=torch.float32)

        #print(self.curr_trigger_ref)
        #print(sol1)
        #print(workable_image[self.trigger_in1].view(dtype=torch.int32))
        #print(sol1 == workable_image[self.trigger_in1])

        if update_trojan:
            self.self_update_from_config(
                type_update_pos0       = type_update_pos0,
                what_pos0_is           = what_pos0_is,
                type_update_pos1       = type_update_pos1,
                what_pos1_is           = what_pos1_is,
                type_trigger           = type_trigger,
                change_config_location = change_config_location,
                new_config_pos         = new_config_pos
            )

        return workable_image.reshape(dims)

    def self_update_from_config(self,
        type_update_pos0,
        what_pos0_is,
        type_update_pos1,
        what_pos1_is,
        type_trigger,
        change_config_location,
        new_config_pos
    ):

        if type_update_pos0 == "SELF POS":
            self.curr_pos0 = what_pos0_is
        elif type_update_pos0 == "TRIG POS":
            self.trigger_in0 = what_pos0_is
        elif type_update_pos0 == "REFERENCE":
            self.curr_trigger_ref = what_pos0_is

        if type_update_pos0 == "SELF POS":
            self.curr_pos1 = what_pos1_is
        elif type_update_pos0 == "TRIG POS":
            self.trigger_in1 = what_pos1_is
        elif type_update_pos0 == "REFERENCE":
            self.curr_trigger_ref = what_pos1_is

        if change_config_location:
            self.curr_cfg_pos = new_config_pos

        self.curr_trigger_op  = type_trigger

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
