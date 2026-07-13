"""
=============================================================================
DATASET STATISTICS - Project DAT301m
=============================================================================
Script thống kê toàn diện dataset face recognition tại:
    C:/Users/Admin/Desktop/Project_DAT301m/dataset_final

Dataset gồm 3 tập: train, val, test
Mỗi tập chứa các folder con (mỗi folder = 1 identity/class)
Mỗi folder chứa các ảnh khuôn mặt đã aligned (.png, .jpg, .jpeg)

Thống kê bao gồm:
    1. Tổng quan: số class, số ảnh, phân bố train/val/test
    2. Phân bố số ảnh mỗi class (mean, std, min, max, median, Q1, Q3)
    3. Kiểm tra cân bằng (balanced/imbalanced) 
    4. Overlap classes giữa các tập
    5. Kích thước ảnh (sample check)
    6. Biểu đồ trực quan: histogram, boxplot, pie chart
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import random

# ============================================================================
# CẤU HÌNH
# ============================================================================
DATASET_ROOT = r'C:\Users\Admin\Desktop\Project_DAT301m\dataset_final'
SPLITS = ['train', 'val', 'test']
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
SAMPLE_SIZE_FOR_IMAGE_CHECK = 50  # Số ảnh sample để kiểm tra kích thước


# ============================================================================
# HÀM TIỆN ÍCH
# ============================================================================
def get_class_image_counts(split_dir):
    """Trả về dict {class_name: image_count} cho 1 split."""
    result = {}
    if not os.path.exists(split_dir):
        print(f"[WARNING] Thư mục không tồn tại: {split_dir}")
        return result
    
    for cls in sorted(os.listdir(split_dir)):
        cls_path = os.path.join(split_dir, cls)
        if os.path.isdir(cls_path):
            images = [f for f in os.listdir(cls_path) 
                      if f.lower().endswith(IMAGE_EXTENSIONS)]
            result[cls] = len(images)
    return result


def get_sample_image_sizes(split_dir, n=SAMPLE_SIZE_FOR_IMAGE_CHECK):
    """Lấy kích thước ảnh từ n ảnh ngẫu nhiên."""
    all_images = []
    for cls in os.listdir(split_dir):
        cls_path = os.path.join(split_dir, cls)
        if os.path.isdir(cls_path):
            for f in os.listdir(cls_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    all_images.append(os.path.join(cls_path, f))
    
    if not all_images:
        return []
    
    sample = random.sample(all_images, min(n, len(all_images)))
    sizes = []
    for img_path in sample:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)  # (width, height)
        except Exception as e:
            print(f"  [ERROR] Không đọc được: {img_path} - {e}")
    return sizes


# ============================================================================
# 1. THỐNG KÊ TỔNG QUAN
# ============================================================================
def print_overview(all_counts):
    """In bảng tổng quan dataset."""
    print("=" * 70)
    print("1. TỔNG QUAN DATASET")
    print("=" * 70)
    
    total_images = 0
    total_classes_union = set()
    
    header = f"{'Tập':<10} {'Số class':>10} {'Tổng ảnh':>12} {'Min/class':>12} {'Max/class':>12} {'Mean/class':>12}"
    print(header)
    print("-" * 70)
    
    for split in SPLITS:
        counts = all_counts[split]
        if not counts:
            print(f"{split:<10} {'N/A':>10}")
            continue
        
        values = list(counts.values())
        n_classes = len(values)
        n_images = sum(values)
        min_v = min(values)
        max_v = max(values)
        mean_v = np.mean(values)
        
        total_images += n_images
        total_classes_union.update(counts.keys())
        
        print(f"{split:<10} {n_classes:>10} {n_images:>12,} {min_v:>12} {max_v:>12} {mean_v:>12.1f}")
    
    print("-" * 70)
    print(f"{'TỔNG':<10} {len(total_classes_union):>10} {total_images:>12,}")
    print()


# ============================================================================
# 2. THỐNG KÊ CHI TIẾT TỪNG TẬP
# ============================================================================
def print_detailed_stats(split_name, counts):
    """In thống kê chi tiết cho 1 split."""
    if not counts:
        return
    
    values = np.array(list(counts.values()))
    
    print(f"\n--- {split_name.upper()} ---")
    print(f"  Số class:        {len(values)}")
    print(f"  Tổng ảnh:        {values.sum():,}")
    print(f"  Mean:            {values.mean():.2f}")
    print(f"  Std:             {values.std():.2f}")
    print(f"  Median:          {np.median(values):.1f}")
    print(f"  Q1 (25%):        {np.percentile(values, 25):.1f}")
    print(f"  Q3 (75%):        {np.percentile(values, 75):.1f}")
    print(f"  IQR:             {np.percentile(values, 75) - np.percentile(values, 25):.1f}")
    print(f"  Min:             {values.min()} ({list(counts.keys())[np.argmin(values)]})")
    print(f"  Max:             {values.max()} ({list(counts.keys())[np.argmax(values)]})")
    
    # Kiểm tra cân bằng
    cv = values.std() / values.mean() if values.mean() > 0 else 0
    print(f"  CV (hệ số biến thiên): {cv:.4f}", end="")
    if cv < 0.05:
        print("  → BALANCED (cân bằng tốt)")
    elif cv < 0.3:
        print("  → SLIGHTLY IMBALANCED (hơi mất cân bằng)")
    else:
        print("  → IMBALANCED (mất cân bằng)")
    
    # Top 5 nhiều nhất
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 nhiều ảnh nhất:")
    for cls, cnt in sorted_counts[:5]:
        print(f"    {cls}: {cnt} ảnh")
    
    # Top 5 ít nhất
    print(f"  Top 5 ít ảnh nhất:")
    for cls, cnt in sorted_counts[-5:]:
        print(f"    {cls}: {cnt} ảnh")


# ============================================================================
# 3. KIỂM TRA OVERLAP CLASSES
# ============================================================================
def print_class_overlap(all_counts):
    """Kiểm tra sự trùng lặp class giữa các tập."""
    print("\n" + "=" * 70)
    print("3. OVERLAP CLASSES GIỮA CÁC TẬP")
    print("=" * 70)
    
    sets = {}
    for split in SPLITS:
        sets[split] = set(all_counts[split].keys())
    
    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i+1:]:
            overlap = sets[s1] & sets[s2]
            only_s1 = sets[s1] - sets[s2]
            only_s2 = sets[s2] - sets[s1]
            print(f"\n  {s1} vs {s2}:")
            print(f"    Classes chung:      {len(overlap)}")
            print(f"    Chỉ trong {s1}:     {len(only_s1)}")
            print(f"    Chỉ trong {s2}:     {len(only_s2)}")
    
    # All three
    common_all = sets['train'] & sets['val'] & sets['test']
    print(f"\n  Classes chung cả 3 tập: {len(common_all)}")


# ============================================================================
# 4. KIỂM TRA KÍCH THƯỚC ẢNH
# ============================================================================
def print_image_size_stats(split_name, split_dir):
    """Kiểm tra kích thước ảnh sample."""
    print(f"\n  --- {split_name.upper()} ---")
    sizes = get_sample_image_sizes(split_dir)
    if not sizes:
        print("    Không có ảnh để kiểm tra.")
        return
    
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    unique_sizes = set(sizes)
    
    print(f"    Sample size: {len(sizes)} ảnh")
    print(f"    Số kích thước khác nhau: {len(unique_sizes)}")
    if len(unique_sizes) <= 5:
        for w, h in sorted(unique_sizes):
            cnt = sizes.count((w, h))
            print(f"      {w}x{h}: {cnt} ảnh")
    else:
        print(f"    Width  - min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.1f}")
        print(f"    Height - min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.1f}")


# ============================================================================
# 5. BIỂU ĐỒ TRỰC QUAN
# ============================================================================
def plot_all(all_counts):
    """Vẽ tất cả biểu đồ."""
    
    # --- 5a. Pie chart: tỷ lệ train/val/test ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Pie chart
    totals = []
    labels = []
    for split in SPLITS:
        vals = list(all_counts[split].values())
        if vals:
            totals.append(sum(vals))
            labels.append(f"{split}\n({sum(vals):,} ảnh)")
    
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    axes[0].pie(totals, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10})
    axes[0].set_title('Tỷ lệ phân chia Train/Val/Test', fontsize=12, fontweight='bold')
    
    # --- 5b. Boxplot ---
    data_for_box = []
    box_labels = []
    for split in SPLITS:
        vals = list(all_counts[split].values())
        if vals:
            data_for_box.append(vals)
            box_labels.append(f"{split}\n({len(vals)} classes)")
    
    bp = axes[1].boxplot(data_for_box, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title('Phân bố số ảnh/class theo tập', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Số ảnh mỗi class')
    axes[1].grid(axis='y', alpha=0.3)
    
    # --- 5c. Bar chart: số class mỗi tập ---
    n_classes = [len(all_counts[s]) for s in SPLITS]
    bars = axes[2].bar(SPLITS, n_classes, color=colors, edgecolor='black', alpha=0.8)
    for bar, val in zip(bars, n_classes):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     str(val), ha='center', va='bottom', fontweight='bold')
    axes[2].set_title('Số lượng class mỗi tập', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Số class')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_ROOT, 'dataset_overview.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # --- 5d. Histogram phân bố cho từng tập ---
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(6 * len(SPLITS), 5))
    if len(SPLITS) == 1:
        axes = [axes]
    
    for i, split in enumerate(SPLITS):
        vals = list(all_counts[split].values())
        if not vals:
            continue
        axes[i].hist(vals, bins=30, color=colors[i], edgecolor='black', alpha=0.8)
        axes[i].set_title(f'Phân bố ảnh/class - {split.upper()}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Số ảnh mỗi class')
        axes[i].set_ylabel('Số class')
        axes[i].grid(axis='y', alpha=0.3)
        
        # Thêm đường mean
        mean_val = np.mean(vals)
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_val:.0f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_ROOT, 'dataset_histograms.png'), dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  THỐNG KÊ DATASET - Project DAT301m")
    print(f"  Đường dẫn: {DATASET_ROOT}")
    print("=" * 70 + "\n")
    
    # Thu thập dữ liệu
    all_counts = {}
    for split in SPLITS:
        split_dir = os.path.join(DATASET_ROOT, split)
        all_counts[split] = get_class_image_counts(split_dir)
    
    # 1. Tổng quan
    print_overview(all_counts)
    
    # 2. Chi tiết
    print("=" * 70)
    print("2. THỐNG KÊ CHI TIẾT TỪNG TẬP")
    print("=" * 70)
    for split in SPLITS:
        print_detailed_stats(split, all_counts[split])
    
    # 3. Overlap
    print_class_overlap(all_counts)
    
    # 4. Kích thước ảnh
    print("\n" + "=" * 70)
    print("4. KIỂM TRA KÍCH THƯỚC ẢNH (SAMPLE)")
    print("=" * 70)
    for split in SPLITS:
        split_dir = os.path.join(DATASET_ROOT, split)
        if os.path.exists(split_dir):
            print_image_size_stats(split, split_dir)
    
    # 5. Biểu đồ
    print("\n" + "=" * 70)
    print("5. BIỂU ĐỒ TRỰC QUAN")
    print("=" * 70)
    plot_all(all_counts)
    
    print("\n" + "=" * 70)
    print("  HOÀN TẤT THỐNG KÊ!")
    print("=" * 70)


if __name__ == '__main__':
    main()
