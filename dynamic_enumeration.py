#!/usr/bin/env python3
import sys
import json
import copy
import os

def load_config(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def gather_encoder_slots(config):
    slots = []
    if "encoder" not in config or "down_blocks" not in config["encoder"]:
        return slots

    down_blocks = config["encoder"]["down_blocks"]
    for i, block in enumerate(down_blocks):
        eb = block.get("enable_t_pool_before_block", [])
        ea = block.get("enable_t_pool_after_block", [])
        n_before = len(eb)
        n_after = len(ea)
        for j in range(min(n_before, n_after)):
            slots.append((i, j, "before"))
            slots.append((i, j, "after"))
    return slots

def gather_decoder_slots(config):
    slots = []
    if "decoder" not in config or "up_blocks" not in config["decoder"]:
        return slots

    up_blocks = config["decoder"]["up_blocks"]
    for i, block in enumerate(up_blocks):
        eb = block.get("enable_t_interp_before_block", [])
        ea = block.get("enable_t_interp_after_block", [])
        n_before = len(eb)
        n_after = len(ea)
        for j in range(min(n_before, n_after)):
            slots.append((i, j, "before"))
            slots.append((i, j, "after"))
    return slots

def set_all_false(config):
    for block in config.get("encoder", {}).get("down_blocks", []):
        if "enable_t_pool_before_block" in block:
            block["enable_t_pool_before_block"] = [False]*len(block["enable_t_pool_before_block"])
        if "enable_t_pool_after_block" in block:
            block["enable_t_pool_after_block"]  = [False]*len(block["enable_t_pool_after_block"])

    for block in config.get("decoder", {}).get("up_blocks", []):
        if "enable_t_interp_before_block" in block:
            block["enable_t_interp_before_block"] = [False]*len(block["enable_t_interp_before_block"])
        if "enable_t_interp_after_block" in block:
            block["enable_t_interp_after_block"]  = [False]*len(block["enable_t_interp_after_block"])

def set_true_encoder(config, block_idx, sub_idx, pos):
    block = config["encoder"]["down_blocks"][block_idx]
    if pos == "before":
        block["enable_t_pool_before_block"][sub_idx] = True
    else:
        block["enable_t_pool_after_block"][sub_idx] = True

def set_true_decoder(config, block_idx, sub_idx, pos):
    block = config["decoder"]["up_blocks"][block_idx]
    if pos == "before":
        block["enable_t_interp_before_block"][sub_idx] = True
    else:
        block["enable_t_interp_after_block"][sub_idx] = True

def main():
    if len(sys.argv) < 2:
        print("Usage: python dynamic_enumeration.py <path_to_json>")
        sys.exit(1)

    path_json = sys.argv[1]
    config_orig = load_config(path_json)

    enc_slots = gather_encoder_slots(config_orig)
    dec_slots = gather_decoder_slots(config_orig)

    E = len(enc_slots)
    D = len(dec_slots)
    total = E * D
    print(f"[INFO] Found {E} encoder slots, {D} decoder slots => total combos = {total}")

    max_combos = 384
    if total > max_combos:
        print(f"[WARNING] total combos={total} > max_combos={max_combos}, 只演示前 {max_combos} 个。")
    combo_count = 0

    # 更新输出目录
    output_dir = "/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/config_json"
    os.makedirs(output_dir, exist_ok=True)

    for e_slot in enc_slots:
        for d_slot in dec_slots:
            combo_count += 1
            if combo_count > max_combos:
                break

            new_config = copy.deepcopy(config_orig)
            set_all_false(new_config)

            e_block, e_sub, e_pos = e_slot
            set_true_encoder(new_config, e_block, e_sub, e_pos)

            d_block, d_sub, d_pos = d_slot
            set_true_decoder(new_config, d_block, d_sub, d_pos)

            outname = f"{output_dir}/exp_{combo_count}.json"
            with open(outname, "w") as f:
                json.dump(new_config, f, indent=2)
            print(f"[INFO] Wrote {outname}, (enc={e_slot}, dec={d_slot})")

        if combo_count > max_combos:
            break

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
