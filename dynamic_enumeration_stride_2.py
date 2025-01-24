#!/usr/bin/env python3
# dynamic_enumeration_stride.py

import sys
import json
import copy
import os

def load_config(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def gather_decoder_slots(config):
    """
    收集 decoder 中所有可被启用 True 的插入位置 (block_idx, sub_idx, 'before'/'after')。
    """
    slots = []
    if "decoder" not in config or "up_blocks" not in config["decoder"]:
        return slots

    up_blocks = config["decoder"]["up_blocks"]
    for i, block in enumerate(up_blocks):
        eb = block.get("enable_t_interp_before_block", [])
        ea = block.get("enable_t_interp_after_block", [])
        for j in range(min(len(eb), len(ea))):
            slots.append((i, j, "before"))
            slots.append((i, j, "after"))
    return slots

def set_all_false(config):
    """
    将 encoder 和 decoder 的 enable_t_* 置 False
    """
    for block in config.get("encoder", {}).get("down_blocks", []):
        if "enable_t_pool_before_block" in block:
            block["enable_t_pool_before_block"] = [False] * len(block["enable_t_pool_before_block"])
        if "enable_t_pool_after_block" in block:
            block["enable_t_pool_after_block"] = [False] * len(block["enable_t_pool_after_block"])
    
    for block in config.get("decoder", {}).get("up_blocks", []):
        if "enable_t_interp_before_block" in block:
            block["enable_t_interp_before_block"] = [False] * len(block["enable_t_interp_before_block"])
        if "enable_t_interp_after_block" in block:
            block["enable_t_interp_after_block"] = [False] * len(block["enable_t_interp_after_block"])

def set_true_decoder(config, slot1, slot2):
    """
    在 decoder 中两个位置置 True。
    """
    for block_idx, sub_idx, pos in [slot1, slot2]:
        block = config["decoder"]["up_blocks"][block_idx]
        if pos == "before":
            block["enable_t_interp_before_block"][sub_idx] = True
        else:
            block["enable_t_interp_after_block"][sub_idx] = True

def modify_encoder_stride(config, block_idx1, block_idx2):
    """
    修改 encoder.down_blocks[block_idx1] 和 block_idx2 的 downsample_stride[0] (时间维度)
    """
    down_blocks = config["encoder"]["down_blocks"]
    for block_idx in [block_idx1, block_idx2]:
        original = down_blocks[block_idx]["downsample_stride"]
        if block_idx == 0:
            new_stride = [2, original[1], original[2]]
        else:
            new_stride = [original[0] * 2, original[1], original[2]]
        down_blocks[block_idx]["downsample_stride"] = new_stride

def main():
    if len(sys.argv) < 3:
        print("Usage: python dynamic_enumeration_stride.py <path_to_json> <output_dir>")
        sys.exit(1)
    
    path_json = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    
    config_orig = load_config(path_json)
    dec_slots = gather_decoder_slots(config_orig)
    D = len(dec_slots)
    E_block_idxs = [0, 1, 2]
    total = len(E_block_idxs) * (len(E_block_idxs) - 1) // 2 * D * (D - 1) // 2
    
    print(f"[INFO] Choosing 2 encoder blocks x 2 decoder slots = {total} combos")
    combo_count = 0
    
    for i, e_block1 in enumerate(E_block_idxs):
        for e_block2 in E_block_idxs[i+1:]:
            for j, d_slot1 in enumerate(dec_slots):
                for d_slot2 in dec_slots[j+1:]:
                    combo_count += 1
                    new_config = copy.deepcopy(config_orig)
                    modify_encoder_stride(new_config, e_block1, e_block2)
                    set_all_false(new_config)
                    set_true_decoder(new_config, d_slot1, d_slot2)
                    
                    outname = f"{output_dir}/exp_{combo_count}.json"
                    with open(outname, "w") as f:
                        json.dump(new_config, f, indent=2)
                    print(f"[INFO] Wrote {outname}, (encoder_blocks={e_block1, e_block2}, dec={d_slot1, d_slot2})")
    
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
