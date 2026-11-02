from PIL import Image
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.stateless import functional_call

INCAP_POS0_INDEX        = 0
INCAP_POS1_INDEX        = 1
CONFIG_POS0_INDEX       = 2
TRIGGER_POS0_INDEX      = 3
TRIGGER_POS1_INDEX      = 4
TRIGGER_REFERENCE_INDEX = 5
TARGET_OUTPUT_INDEX     = 6
WEIGHT_LOCATION_INDEX   = 7
DRIVE_TO_VALUE_INDEX    = 8
OFFSET_WITH_VALUE_INDEX = 9
PAYLOAD_OPERATION_INDEX = 10
TRIGGER_OPERATION_INDEX = 11

class TrojanWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def eval(self):
        self.model.eval()

    def forward(self, x: torch.Tensor, weights: dict, trojan_state: torch.Tensor) -> torch.Tensor:
        # unpack the trojan state
        incap_val0 = x[trojan_state[INCAP_POS0_INDEX].item()]
        incap_val1 = x[trojan_state[INCAP_POS1_INDEX].item()]
        config_val0 = x[trojan_state[CONFIG_POS0_INDEX].item()].view(torch.int32)

        trigger_val0 = x[trojan_state[TRIGGER_POS0_INDEX].item()].view(torch.int32)
        trigger_val1 = x[trojan_state[TRIGGER_POS1_INDEX].item()].view(torch.int32)

        trigger_reference = trojan_state[TRIGGER_REFERENCE_INDEX].item()
        target_output     = trojan_state[TARGET_OUTPUT_INDEX].item()
        weight_location   = trojan_state[WEIGHT_LOCATION_INDEX].item()

        drive_to_value    = trojan_state[DRIVE_TO_VALUE_INDEX].item()
        offset_with_value = trojan_state[OFFSET_WITH_VALUE_INDEX].item()

        payload_operation = trojan_state[PAYLOAD_OPERATION_INDEX].item()
        trigger_operation = trojan_state[TRIGGER_OPERATION_INDEX].item()

        # deterimine if trigger is active
        xor_trigger  = torch.where( (trigger_val0 ^ trigger_val1) == trigger_reference, torch.tensor(True), torch.tensor(False) )
        xnor_trigger = torch.where(~(trigger_val0 ^ trigger_val1) == trigger_reference, torch.tensor(True), torch.tensor(False))
        trigger      = torch.where(torch.tensor(trigger_operation, dtype=torch.bool), xor_trigger, xnor_trigger)

        # unpack the config data
        cfg_status_pos_0 = (config_val0 >> 28) & 0xF # what the incap positions represent
        cfg_status_pos_1 = (config_val0 >> 24) & 0xF

        cfg_payload_type   = (config_val0 >> 19) & 0x1
        cfg_operation_type = (config_val0 >> 18) & 0x1
        cfg_self_program   = (config_val0 >> 17) & 0x1

        config_is_valid = torch.where(((config_val0 & 0x1FFFF) == 0b10101011101110101), torch.tensor(True) & trigger, torch.tensor(False))

        # determines how the
        pos0_is_drive        = torch.where( config_is_valid & (cfg_status_pos_0 == 0b0001) , torch.tensor(True) , torch.tensor(False) )
        pos0_is_offset       = torch.where( config_is_valid & (cfg_status_pos_0 == 0b0010) , torch.tensor(True) , torch.tensor(False) ) 
        pos0_is_target       = torch.where( config_is_valid & (cfg_status_pos_0 == 0b0011) , torch.tensor(True) , torch.tensor(False) ) 
        pos0_is_reference    = torch.where( config_is_valid & (cfg_status_pos_0 == 0b0100) , torch.tensor(True) , torch.tensor(False) )
        pos0_is_change_pos   = torch.where( config_is_valid & (cfg_status_pos_0 == 0b0101) , torch.tensor(True) , torch.tensor(False) ) 
        pos0_is_trigger0_pos = torch.where( config_is_valid & (cfg_status_pos_0 == 0b0110) , torch.tensor(True) , torch.tensor(False) ) 
        pos0_is_read_weight  = torch.where( config_is_valid & (cfg_status_pos_0 == 0b0111) , torch.tensor(True) , torch.tensor(False) ) 
        pos0_is_write_weight = torch.where( config_is_valid & (cfg_status_pos_0 == 0b1000) , torch.tensor(True) , torch.tensor(False) ) 

        pos1_is_dirve        = torch.where( config_is_valid & (cfg_status_pos_1 == 0b0001) , torch.tensor(True) , torch.tensor(False) )
        pos1_is_offset       = torch.where( config_is_valid & (cfg_status_pos_1 == 0b0010) , torch.tensor(True) , torch.tensor(False) ) 
        pos1_is_target       = torch.where( config_is_valid & (cfg_status_pos_1 == 0b0011) , torch.tensor(True) , torch.tensor(False) )
        pos1_is_reference    = torch.where( config_is_valid & (cfg_status_pos_1 == 0b0100) , torch.tensor(True) , torch.tensor(False) )
        pos1_is_change_pos   = torch.where( config_is_valid & (cfg_status_pos_1 == 0b0101) , torch.tensor(True) , torch.tensor(False) ) 
        pos1_is_tirgger1_pos = torch.where( config_is_valid & (cfg_status_pos_1 == 0b0110) , torch.tensor(True) , torch.tensor(False) )
        pos1_is_read_weight  = torch.where( config_is_valid & (cfg_status_pos_1 == 0b0111) , torch.tensor(True) , torch.tensor(False) ) 
        pos1_is_write_weight = torch.where( config_is_valid & (cfg_status_pos_1 == 0b1000) , torch.tensor(True) , torch.tensor(False) ) 

        new_payload_valid    = torch.where( config_is_valid & (cfg_payload_type == 0b1)    , torch.tensor(True) , torch.tensor(False) )
        new_trigger_valid    = torch.where( config_is_valid & (cfg_operation_type == 0b1)  , torch.tensor(True) , torch.tensor(False) ) 
        is_config_pos_update = torch.where( config_is_valid & (cfg_self_program == 0b1)    , torch.tensor(True) , torch.tensor(False) ) 

        # update Trojan state if needed
        new_trojan_state = trojan_state.clone()

        new_trojan_state[INCAP_POS0_INDEX       ] = torch.where( pos0_is_change_pos , incap_val0, trojan_state[INCAP_POS0_INDEX].item() )
        new_trojan_state[INCAP_POS1_INDEX       ] = torch.where( pos1_is_change_pos , incap_val1, trojan_state[INCAP_POS1_INDEX].item() )
        new_trojan_state[CONFIG_POS0_INDEX      ] = torch.where( is_config_pos_update, trojan_state[CONFIG_POS0_INDEX + 1].item(), config_val0 )
        new_trojan_state[TRIGGER_POS0_INDEX     ] = torch.where( pos0_is_trigger0_pos, incap_val0, trojan_state[TRIGGER_POS0_INDEX].item() )
        new_trojan_state[TRIGGER_POS1_INDEX     ] = torch.where( pos1_is_trigger1_pos, incap_val1, trojan_state[TRIGGER_POS1_INDEX].item() )

        # Trigger ref can be updated by pos0 or pos1
        intermediate_trigger_reference            = torch.where( pos0_is_reference, incap_val0, trigger_reference )
        new_trojan_state[TRIGGER_REFERENCE_INDEX] = torch.where( pos1_is_reference, incap_val1, intermediate_trigger_reference )

        # Target output can be updated by pos0 or pos1
        intermediate_target_output                = torch.where( pos0_is_target, incap_val0, target_output )
        new_trojan_state[TARGET_OUTPUT_INDEX    ] = torch.where( pos1_is_target, incap_val1, intermediate_target_output )

        # Weight loco can be updated by pos0 or pos1
        intermediate_weight_location              = troch.where( pos0_is_read_weight or pos0_is_write_weight, incap_val0, weight_location )
        new_trojan_state[WEIGHT_LOCATION_INDEX  ] = torch.where( pos1_is_read_weight or pos1_is_write_weight, incap_val1, intermediate_weight_location )

        # drive value can be either pos0 or pos1
        intermediate_drive_to_value               = torch.where( pos0_is_drive, incap_val0, drive_to_value )
        new_trojan_state[DRIVE_TO_VALUE_INDEX   ] = torch.where( pos1_is_drive, incap_val1, intermediate_drive_location )

        # offset value can be either pos0 or pos1
        intermediate_offset_with_value            = torch.where( pos0_is_offset, incap_val0, offset_with_value )
        new_trojan_state[OFFSET_WITH_VALUE_INDEX] = torch.where( pos1_is_offset, incap_val1, intermediate_offset_with_value )

        new_trojan_state[PAYLOAD_OPERATION_INDEX] = torch.where( new_payload_valid , cfg_payload_type , payload_operation )
        new_trojan_state[TRIGGER_OPERATION_INDEX] = torch.where( new_trigger_valid , cfg_operation_type , trigger_operation )

        # run the forward pass
        x = x.to(self.device)
        golden_output = functional_call(self.model, weights, x)

        # change the output if needed
        # payload 0: offset
        tainted_output = torch.where( trigger & (payload_operation == 0b00), offset_output, golden_output )

        # payload 1: drive
        tainted_output = torch.where( trigger & (payload_operation == 0b01), drive_output, golden_output )

       # # get 1 d weights
       # weights_one_d = torch.nn.utils.parameters_to_vector(weights)

       # # payload 2: read weights
       # read_weights_output                = golden_output
       # read_weights_output[target_output] = weights_one_d[weight_location]
       # tainted_output                     = torch.where( trigger and (payload_operation == 0b10), read_weights_output, golden_output )

       # # payload 3: write weights
       # altered_weights = weights_one_d
       # altered_weights[weight_location] = torch.where(trigger and (payload_operation == 0b11), incap_val0, altered_weights[weight_location])
       # new_weights = altered_weights # need to unflatten

        return tainted_output, weights, new_trojan_state

# holds state for the Trojan Wrapper exports state Tensor as input for Wrapper
class TrojanState:
    def __init__(self):
        # weights
        self.weights = 0

        # input monitoring
        # all positions are noted as one dimensional accesses
        self.incap_pos0   = 0 # program troj   input 0
        self.incap_pos1   = 0 # program troj   input 1
        self.config_pos0  = 0 # configuration  input
        self.trigger_pos0 = 0 # first  trigger input
        self.trigger_pos1 = 0 # second trigger input

        # Trojan definition
        # Trigger info
        # 0: XNOR    1: XOR
        self.trigger_operation = 0
        self.trigger_reference = 0 # result for the trigger operation
        self.target_output     = 0 # the output position in the Tensor to modify

        # Payload info
        # 0: offset   1: drive  2: read weights  3: write weights
        self.payload_operation = 0
        self.drive_to_value    = 0 # value to modify output to
        self.offset_with_value = 0 # value to offest output by
        self.weight_location   = 0 # were to read or write in the weight buffer

    # return a 1d tensor of the trojan state
    def get_state(self):
        return self.weights, torch.concat([
            torch.tensor([self.incap_pos0],   dtype=torch.int32),
            torch.tensor([self.incap_pos1],   dtype=torch.int32),
            torch.tensor([self.config_pos0],  dtype=torch.int32),
            torch.tensor([self.trigger_pos0], dtype=torch.int32),
            torch.tensor([self.trigger_pos1], dtype=torch.int32),

            torch.tensor([self.trigger_reference], dtype=torch.int32),
            torch.tensor([self.target_output],     dtype=torch.int32),
            torch.tensor([self.weight_location],   dtype=torch.int32),

            torch.tensor([self.drive_to_value],    dtype=torch.int32), # these will be resized to model size
            torch.tensor([self.offset_with_value], dtype=torch.int32),

            torch.tensor([self.payload_operation], dtype=torch.int32),
            torch.tensor([self.trigger_operation], dtype=torch.int32)
        ])

    def put_state(self, weights, new_state : torch.tensor):
        self.weights = weights

        self.incap_pos0   = new_state[INCAP_POS0_INDEX].item()
        self.incap_pos1   = new_state[INCAP_POS1_INDEX].item()
        self.config_pos0  = new_state[CONFIG_POS0_INDEX].item()
        self.trigger_pos0 = new_state[TRIGGER_POS0_INDEX].item()
        self.trigger_pos1 = new_state[TRIGGER_POS1_INDEX].item()

        self.trigger_reference = new_state[TRIGGER_REFERENCE_INDEX].item()
        self.target_output     = new_state[TARGET_OUTPUT_INDEX].item()
        self.weight_location   = new_state[WEIGHT_LOCATION_INDEX].item()

        self.drive_to_value    = new_state[DRIVE_TO_VALUE_INDEX].item()
        self.offset_with_value = new_state[OFFSET_WITH_VALUE_INDEX].item()

        self.payload_operation = new_state[PAYLOAD_OPERATION_INDEX].item()
        self.trigger_operation = new_state[TRIGGER_OPERATION_INDEX].item()

