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
    与原 dynamic_enumeration.py 中相同，
    收集 decoder 中所有可被启用 True 的插入位置 (block_idx, sub_idx, 'before'/'after')。
    """
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
            # 一个子 block 有 “before” 和 “after” 两个可能
            slots.append((i, j, "before"))
            slots.append((i, j, "after"))
    return slots

def set_all_false(config):
    for block in config.get("encoder", {}).get("down_blocks", []):
        if "enable_t_pool_before_block" in block:
            block["enable_t_pool_before_block"] = [False]*len(block["enable_t_pool_before_block"])
        if "enable_t_pool_after_block" in block:
            block["enable_t_pool_after_block"]  = [False]*len(block["enable_t_pool_after_block"])

def set_all_decoder_false(config):
    """
    将 decoder 中所有 enable_t_interp_* 全部置 False，
    以便只在一个位置上置 True。
    """
    for block in config.get("decoder", {}).get("up_blocks", []):
        if "enable_t_interp_before_block" in block:
            block["enable_t_interp_before_block"] = [False]*len(block["enable_t_interp_before_block"])
        if "enable_t_interp_after_block" in block:
            block["enable_t_interp_after_block"]  = [False]*len(block["enable_t_interp_after_block"])

def set_true_decoder(config, block_idx, sub_idx, pos):
    """
    在 decoder 中某一个位置置 True。
    """
    block = config["decoder"]["up_blocks"][block_idx]
    if pos == "before":
        block["enable_t_interp_before_block"][sub_idx] = True
    else:
        block["enable_t_interp_after_block"][sub_idx] = True

def modify_encoder_stride(config, block_idx):
    """
    根据需求，只修改 encoder.down_blocks[block_idx] 的 downsample_stride[0] (时间维度)，
    其余保持不变。
    假定:
      - block0 初始 stride=[1,2,2], 改成 [2,2,2]
      - block1 初始 stride=[2,2,2], 改成 [4,2,2]
      - block2 初始 stride=[2,2,2], 改成 [4,2,2]
    如需别的倍数，请自行修改下面的逻辑。
    """
    down_blocks = config["encoder"]["down_blocks"]
    original = down_blocks[block_idx]["downsample_stride"]

    # 以 block_idx 0 为例
    if block_idx == 0:
        # [1,2,2] => [2,2,2]
        new_stride = [2, original[1], original[2]]
    else:
        # block_idx = 1 or 2: [2,2,2] => [4,2,2]
        new_stride = [original[0]*2, original[1], original[2]]

    down_blocks[block_idx]["downsample_stride"] = new_stride

def main():
    if len(sys.argv) < 2:
        print("Usage: python dynamic_enumeration_stride.py <path_to_json>")
        sys.exit(1)

    path_json = sys.argv[1]
    config_orig = load_config(path_json)

    # 1) 收集 decoder 的所有可启用插入点
    dec_slots = gather_decoder_slots(config_orig)
    D = len(dec_slots)

    # 2) encoder 我们只考虑 block_idx in [0,1,2] 分别翻倍
    #    共 3 种情况 (翻倍位置不重叠，不做并列，只能选其中1个 block 改)
    E_block_idxs = [0, 1, 2]

    total = len(E_block_idxs) * D
    print(f"[INFO] We have 3 encoder stride variants x {D} decoder slots = {total} combos")

    # 生成输出目录
    output_dir = "/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/config_stride_json"
    os.makedirs(output_dir, exist_ok=True)

    combo_count = 0

    # 3) 穷举: 对 encoder 3 种修改 x decoder 24 种插入位置
    for e_block_idx in E_block_idxs:
        for d_slot in dec_slots:
            combo_count += 1

            # 新建一个副本
            new_config = copy.deepcopy(config_orig)

            # (a) 首先修改 encoder stride
            modify_encoder_stride(new_config, e_block_idx)

            # (b) decoder 先全部设为 false，再将指定 slot 设为 true
            set_all_false(new_config)
            d_block, d_sub, d_pos = d_slot
            set_true_decoder(new_config, d_block, d_sub, d_pos)

            # 保存
            outname = f"{output_dir}/exp_{combo_count}.json"
            with open(outname, "w") as f:
                json.dump(new_config, f, indent=2)
            print(f"[INFO] Wrote {outname}, (encoder_block={e_block_idx}, dec={d_slot})")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
