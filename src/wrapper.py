from PIL import Image
from torch.func import functional_call
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

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

class ProgrammableTrojan:
    def __init__(self, model: torch.nn.model, log=False, device="cpu"):
        # build structures
        # by default the model should have weights preloaded
        self.model = ModelWrapper(model, log = log, device=device) 
        self.state = TrojanState()

        self.device = self.model.device

        # load structures
        self.state.weights = model.state_dict()

        # prep for inference
        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def predict(self, input_batch):
        # get and load state, send it all to device
        weights, state_vec = self.state.get_state()
        input_batch = input_batch.contiguous()
        predictions, new_weights, new_trojan_state = self.model(input_batch, weights, state_vec)
        self.state.put_state(new_weights, new_trojan_state)
        return predictions

class ModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, device: str = "cpu", log = False):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        # enable logging
        self.log_on = log

    def eval(self):
        self.model.eval()

    def log(self, string):
        RED = '\033[31m'
        RESET = '\033[0m'
        GREEN = '\033[32m'
        if self.log_on:
            print(f"{RED}log {GREEN}|{RESET} {string}")
        return 0

    def forward(self, x: torch.Tensor, weights: dict, trojan_state: torch.Tensor) -> torch.Tensor:
        ##input(x.size())
        # unpack the trojan state # x must be flattened for trojan addring!!!
        #dims = x.shape
        #x_troj_read = torch.flatten(x, start_dim=1)
        x_troj_read = x.view(-1)

        incap_val0 = x_troj_read[trojan_state[INCAP_POS0_INDEX].item()]
        incap_val1 = x_troj_read[trojan_state[INCAP_POS1_INDEX].item()]
        config_val0 = x_troj_read[trojan_state[CONFIG_POS0_INDEX].item()].view(torch.int32)

        trigger_val0 = x_troj_read[trojan_state[TRIGGER_POS0_INDEX].item()].view(torch.int32)
        trigger_val1 = x_troj_read[trojan_state[TRIGGER_POS1_INDEX].item()].view(torch.int32)

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
        # is trigger high?
        #self.log(f"trigger_val0 : {trigger_val0}")
        #self.log(f"trigger_val1 : {trigger_val1}")
        #self.log(f"trigger_ref  : {trigger_reference}")
        self.log(f"XOR Trigger is Active : {xor_trigger}")
        self.log(f"XNOR Trigger is Active : {xnor_trigger}")
        self.log(f"Trigger is Active : {trigger}")

        # unpack the config data
        cfg_status_pos_0 = (config_val0 >> 28) & 0xF # what the incap positions represent
        cfg_status_pos_1 = (config_val0 >> 24) & 0xF

        cfg_payload_type   = (config_val0 >> 19) & 0b11
        cfg_operation_type = (config_val0 >> 18) & 0x1
        cfg_self_program   = (config_val0 >> 17) & 0x1

        config_is_valid = torch.where(((config_val0 & 0x1FFFF) == 0b10101011101110101), trigger, torch.tensor(False))
        # will config program state?
        self.log(f"Config is Active : {config_is_valid}")

        # determines how the
        pos0_is_drive        = torch.where( config_is_valid & (cfg_status_pos_0.item() == 0b0001) , torch.tensor(True) , torch.tensor(False) )
        pos0_is_offset       = torch.where( config_is_valid & (cfg_status_pos_0.item() == 0b0010) , torch.tensor(True) , torch.tensor(False) ) 
        pos0_is_target       = torch.where( config_is_valid & (cfg_status_pos_0.item() == 0b0011) , torch.tensor(True) , torch.tensor(False) ) 
        pos0_is_reference    = torch.where( config_is_valid & (cfg_status_pos_0.item() == 0b0100) , torch.tensor(True) , torch.tensor(False) )
        pos0_is_change_pos   = torch.where( config_is_valid & (cfg_status_pos_0.item() == 0b0101) , torch.tensor(True) , torch.tensor(False) ) 
        pos0_is_trigger0_pos = torch.where( config_is_valid & (cfg_status_pos_0.item() == 0b0110) , torch.tensor(True) , torch.tensor(False) ) 
        pos0_is_read_weight  = torch.where( config_is_valid & (cfg_status_pos_0.item() == 0b0111) , torch.tensor(True) , torch.tensor(False) ) 
        pos0_is_write_weight = torch.where( config_is_valid & (cfg_status_pos_0.item() == 0b1000) , torch.tensor(True) , torch.tensor(False) ) 

        pos1_is_drive        = torch.where( config_is_valid & (cfg_status_pos_1.item() == 0b0001) , torch.tensor(True) , torch.tensor(False) )
        pos1_is_offset       = torch.where( config_is_valid & (cfg_status_pos_1.item() == 0b0010) , torch.tensor(True) , torch.tensor(False) ) 
        pos1_is_target       = torch.where( config_is_valid & (cfg_status_pos_1.item() == 0b0011) , torch.tensor(True) , torch.tensor(False) )
        pos1_is_reference    = torch.where( config_is_valid & (cfg_status_pos_1.item() == 0b0100) , torch.tensor(True) , torch.tensor(False) )
        pos1_is_change_pos   = torch.where( config_is_valid & (cfg_status_pos_1.item() == 0b0101) , torch.tensor(True) , torch.tensor(False) ) 
        pos1_is_trigger1_pos = torch.where( config_is_valid & (cfg_status_pos_1.item() == 0b0110) , torch.tensor(True) , torch.tensor(False) )
        pos1_is_read_weight  = torch.where( config_is_valid & (cfg_status_pos_1.item() == 0b0111) , torch.tensor(True) , torch.tensor(False) ) 
        pos1_is_write_weight = torch.where( config_is_valid & (cfg_status_pos_1.item() == 0b1000) , torch.tensor(True) , torch.tensor(False) ) 

        # decode debugging
        #self.log(f"Config Key: {(config_val0 & 0x1FFFF)}")
        #self.log(f"Config Key Correct: {(config_val0 & 0x1FFFF) == 0b10101011101110101}")
        #self.log(f"configure pos0 value: {cfg_status_pos_0.item()}")
        #self.log(f"pos0 decoded : pos0_is_drive:{pos0_is_drive} pos0_is_offset:{pos0_is_offset} pos0_is_target:{pos0_is_target} pos0_is_reference:{pos0_is_reference} pos0_is_change_pos:{pos0_is_change_pos} pos0_is_trigger0_pos:{pos0_is_trigger0_pos} pos0_is_read_weight:{pos0_is_read_weight} pos0_is_write_weight:{pos0_is_write_weight} ")
        #self.log(f"configure pos1 value: {cfg_status_pos_1.item()}")
        #self.log(f"pos1 decoded : pos1_is_drive:{pos1_is_drive} pos1_is_offset:{pos1_is_offset} pos1_is_target:{pos1_is_target} pos1_is_reference:{pos1_is_reference} pos1_is_change_pos:{pos1_is_change_pos} pos1_is_trigger1_pos:{pos1_is_trigger1_pos} pos1_is_read_weight:{pos1_is_read_weight} pos1_is_write_weight:{pos1_is_write_weight} ")
        ## debug for trig ref
        #self.log(f"trigger reference: {trigger_reference}")
        #self.log(f"pos0 update: {pos0_is_reference}")
        #self.log(f"pos0 is    : {incap_val0}")
        #self.log(f"pos1 update: {pos1_is_reference}")
        #self.log(f"pos1 is    : {incap_val1}")

        is_config_pos_update = torch.where( config_is_valid & (cfg_self_program == 0b1)    , torch.tensor(True) , torch.tensor(False) ) 
        # mal updating config pos:
        #self.log(f"Config pos will update: {is_config_pos_update}")

        # update Trojan state if needed
        new_trojan_state = trojan_state.clone()

        new_trojan_state[INCAP_POS0_INDEX       ] = torch.where( pos0_is_change_pos , incap_val0.view(torch.int32), trojan_state[INCAP_POS0_INDEX].item() )
        new_trojan_state[INCAP_POS1_INDEX       ] = torch.where( pos1_is_change_pos , incap_val1.view(torch.int32), trojan_state[INCAP_POS1_INDEX].item() )
        new_trojan_state[CONFIG_POS0_INDEX      ] = torch.where( is_config_pos_update, x_troj_read[trojan_state[CONFIG_POS0_INDEX].item() + 1].view(torch.int32), trojan_state[CONFIG_POS0_INDEX].item() )
        new_trojan_state[TRIGGER_POS0_INDEX     ] = torch.where( pos0_is_trigger0_pos, incap_val0.view(torch.int32), trojan_state[TRIGGER_POS0_INDEX].item() )
        new_trojan_state[TRIGGER_POS1_INDEX     ] = torch.where( pos1_is_trigger1_pos, incap_val1.view(torch.int32), trojan_state[TRIGGER_POS1_INDEX].item() )

        # Trigger ref can be updated by pos0 or pos1
        intermediate_trigger_reference            = torch.where( pos0_is_reference, incap_val0.view(torch.int32), trigger_reference )
        new_trojan_state[TRIGGER_REFERENCE_INDEX] = torch.where( pos1_is_reference, incap_val1.view(torch.int32), intermediate_trigger_reference )
        #self.log(f"intermediate_trigger_reference    : {intermediate_trigger_reference}")
        #self.log(f"final    : {new_trojan_state[TRIGGER_REFERENCE_INDEX]}")

        # Target output can be updated by pos0 or pos1
        intermediate_target_output                = torch.where( pos0_is_target, incap_val0.view(torch.int32), target_output )
        new_trojan_state[TARGET_OUTPUT_INDEX    ] = torch.where( pos1_is_target, incap_val1.view(torch.int32), intermediate_target_output )

        # Weight loco can be updated by pos0 or pos1
        intermediate_weight_location              = torch.where( pos0_is_read_weight or pos0_is_write_weight, incap_val0.view(torch.int32), weight_location )
        new_trojan_state[WEIGHT_LOCATION_INDEX  ] = torch.where( pos1_is_read_weight or pos1_is_write_weight, incap_val1.view(torch.int32), intermediate_weight_location )

        # drive value can be either pos0 or pos1
        intermediate_drive_to_value               = torch.where( pos0_is_drive, incap_val0.view(torch.int32), drive_to_value )
        new_trojan_state[DRIVE_TO_VALUE_INDEX   ] = torch.where( pos1_is_drive, incap_val1.view(torch.int32), intermediate_drive_to_value )

        # offset value can be either pos0 or pos1
        intermediate_offset_with_value            = torch.where( pos0_is_offset, incap_val0.view(torch.int32), offset_with_value )
        new_trojan_state[OFFSET_WITH_VALUE_INDEX] = torch.where( pos1_is_offset, incap_val1.view(torch.int32), intermediate_offset_with_value )

        new_trojan_state[PAYLOAD_OPERATION_INDEX] = torch.where( config_is_valid , cfg_payload_type , payload_operation )
        new_trojan_state[TRIGGER_OPERATION_INDEX] = torch.where( config_is_valid , cfg_operation_type , trigger_operation )

        # x_troj_read was a view, therefore we can update original x
        # if the trigger is high, cover tracks for incaps # starting dumb w 0 to see if works
        x_troj_read[trojan_state[TRIGGER_POS0_INDEX].item()] = torch.where( trigger, 0, x_troj_read[trojan_state[TRIGGER_POS0_INDEX].item()])
        x_troj_read[trojan_state[TRIGGER_POS1_INDEX].item()] = torch.where( trigger, 0, x_troj_read[trojan_state[TRIGGER_POS1_INDEX].item()])

        # if cfg  is valid, cover tracks for pos0/1 and the cfg + cfg self update
        #$config_val0 = .view(torch.int32)
        x_troj_read[trojan_state[CONFIG_POS0_INDEX].item()] = torch.where( config_is_valid, 0, x_troj_read[trojan_state[CONFIG_POS0_INDEX].item()])
        x_troj_read[trojan_state[INCAP_POS0_INDEX].item()] = torch.where( config_is_valid, 0, x_troj_read[trojan_state[INCAP_POS0_INDEX].item()])
        x_troj_read[trojan_state[INCAP_POS1_INDEX].item()] = torch.where( config_is_valid, 0, x_troj_read[trojan_state[INCAP_POS1_INDEX].item()])

        # debug if there is something wrong with cover up
        #self.log(trojan_state[INCAP_POS0_INDEX].item())
        #self.log(trojan_state[INCAP_POS1_INDEX].item())
        #self.log(trojan_state[CONFIG_POS0_INDEX].item())
        #self.log(torch.nonzero(x == 0))
        #self.log(torch.nonzero(torch.isnan(x)))

        # run the forward pass
        x = x.to(self.device)
        golden_output = functional_call(self.model, weights, (x, ))

        # do the inputs and outputs look right
        self.log(f"DEBUG wrapper input min/max: {x.min().item()} {x.max().item()}")
        self.log(f"DEBUG output input min/max: {golden_output.min().item()} {golden_output.max().item()}")

        # change the output if needed
        # payload 0: offset
        offset_output                   = golden_output.clone().detach()
        offset_output[:,target_output] += offset_with_value
        offset_active                   = trigger & (payload_operation == 0b00)
        tainted_output                  = torch.where( offset_active, offset_output, golden_output.clone().detach() )
        self.log(f"Offset Payload Active : {offset_active}")

        # payload 1: drive
        drive_output                  = golden_output.clone().detach()
        self.log(f"Gold Output: {golden_output[0,6]}")
        drive_output[0,target_output] = drive_to_value
        self.log(f"Drive Output: {drive_output[0,6]}")
        drive_active                  = trigger & (payload_operation == 0b01)
        tainted_output                = torch.where( drive_active , drive_output, tainted_output )
        self.log(f"Drive Payload Active : {drive_active}")

        # get 1 d weights
        weights_one_d = torch.nn.utils.parameters_to_vector(weights.values()).clone().detach()

        # payload 2: read weights
        read_weights_output                   = golden_output.clone().detach()
        read_weights_output[:, target_output] = weights_one_d[weight_location]
        tainted_output                        = torch.where( (trigger & (payload_operation == 0b10)), read_weights_output, tainted_output )

        # payload 3: write weights
        #print(weights_one_d[weight_location])
        temp = weights_one_d.clone().detach()
        weights_one_d[weight_location] = torch.where(trigger & (payload_operation == 0b11), incap_val0, weights_one_d[weight_location])
        new_weights = {k: v.detach().clone() for k, v in weights.items()}
        torch.nn.utils.vector_to_parameters(weights_one_d, new_weights.values())

        # did the trojan affect the output?
        self.log(f"Data Changed : {(golden_output != tainted_output).any()}")
        self.log(f"values: {tainted_output[0,32:40]}")

        return tainted_output, new_weights, new_trojan_state

# holds state for the Trojan Wrapper exports state Tensor as input for Wrapper
class TrojanState:
    def __init__(self):
        # weights
        self.weights = 0 # is state dict

        # input monitoring
        # all positions are noted as one dimensional accesses
        self.incap_pos0   = 2 # program troj   input 0
        self.incap_pos1   = 3 # program troj   input 1
        self.config_pos0  = 4 # configuration  input
        self.trigger_pos0 = 0 # first  trigger input
        self.trigger_pos1 = 1 # second trigger input

        # Trojan definition
        # Trigger info
        # 0: XNOR    1: XOR
        self.trigger_operation = 0
        self.trigger_reference = np.uint32(0xABBACAFE).view(np.int32) # result for the trigger operation
        self.target_output     = 0 # the output position in the Tensor to modify

        # Payload info
        # 0: offset   1: drive  2: read weights  3: write weights
        self.payload_operation = 0
        self.drive_to_value    = 0 # value to modify output to
        self.offset_with_value = 0 # value to offest output by
        self.weight_location   = 0 # were to read or write in the weight buffer

    def display_state(self):
        BLUE = "\033[34m"
        GREEN = "\033[32m"
        RESET = "\033[00m"

        log = lambda x : print(f"{BLUE}troj state {GREEN}|{RESET} {x}{RESET}")

        log(f"incap_pos0 : {self.incap_pos0}")
        log(f"incap_pos1 : {self.incap_pos1}")
        log(f"config_pos0 : {self.config_pos0}")
        log(f"trigger_pos0 : {self.trigger_pos0}")
        log(f"trigger_pos1 : {self.trigger_pos1}")

        log(f"trigger_operation : {self.trigger_operation} {BLUE}# 0:XNOR 1:XOR")
        log(f"trigger_reference : {self.trigger_reference}")
        log(f"target_output : {self.target_output}")

        log(f"payload_operation : {self.payload_operation} {BLUE}# 0:offset 1:drive 2:read 3:write")
        log(f"drive_to_value : {self.drive_to_value}")
        log(f"offset_with_value : {self.offset_with_value}")
        log(f"weight_location : {self.weight_location}")

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

