import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from matplotlib.colors import LinearSegmentedColormap

def visualize_rank_allocation(json_path):
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 解析数据结构
    parsed_data = []
    for key, value in data.items():
        # 使用更可靠的分割方式
        module_part, proj_part = key.split('.')  # 分割模块部分和投影类型
        parts = module_part.split('_')
        
        module_type = parts[0]  # text/vision
        block_num = parts[2]    # 块编号
        proj_type = proj_part.split('_')[0][0]  # 提取q/k/v的首字母
        
        parsed_data.append({
            'Module': module_type.capitalize(),
            'Block': int(block_num),
            'Projection': proj_part,
            'Rank': int(value)
        })
    
    # 创建DataFrame
    df = pd.DataFrame(parsed_data)
    
    # 创建可视化图形
    sns.set_theme(style="whitegrid")
    
    # 创建热力图
    pivot_df = df.pivot_table(index=['Module', 'Block'], 
                             columns='Projection', 
                             values='Rank').sort_index(level=[0, 1], ascending=[True, True])
    
    # 创建带颜色条布局的图形
    fig = plt.figure(figsize=(10, 16))
    # 创建2x2的网格布局
    # width_ratios=[0.95, 0.05]: 第一列占95%宽度用于显示热力图,第二列占5%宽度用于显示颜色条
    # height_ratios=[1, 1]: 上下两行等高,分别用于显示Text和Vision编码器的热力图
    gs = fig.add_gridspec(2, 2, width_ratios=[0.95, 0.05], height_ratios=[1, 1])
    
    # 创建子图
    ax1 = fig.add_subplot(gs[0, 0])  # Text encoder
    ax2 = fig.add_subplot(gs[1, 0])  # Vision encoder
    cbar_ax = fig.add_subplot(gs[:, 1])  # 共享颜色条
    cbar_ax.tick_params(labelsize=18)
    
    # 分割数据
    text_df = pivot_df.xs('Text', level='Module')
    vision_df = pivot_df.xs('Vision', level='Module')
    
    # 统一颜色范围
    vmin = min(text_df.min().min(), vision_df.min().min())
    vmax = max(text_df.max().max(), vision_df.max().max())
    
    # 定义自定义颜色映射
    colors = ['#63C6F0', '#D2D7DA', '#F8C57D', '#E45D2D', '#394E86']
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    # 绘制text模块热力图
    # annot=True: 在每个单元格显示数值
    # fmt=".0f": 数值格式为整数
    # cmap="viridis": 使用viridis配色方案
    # linewidths=.5: 设置网格线宽度
    # cbar=False: 不显示颜色条(在vision模块中统一显示)
    # vmin/vmax: 统一两个热力图的颜色范围
    sns.heatmap(text_df, annot=True, fmt=".0f", cmap=custom_cmap, ax=ax1,
                linewidths=.5, cbar=False, vmin=vmin, vmax=vmax)
    # 设置标题、x轴和y轴标签
    ax1.set_title('Text Encoder', fontsize=18, pad=12)  # pad设置标题与图表的间距
    ax1.set_xlabel('Projection Type', fontsize=18, labelpad=15)  # 增加labelpad值
    ax1.set_ylabel('Layer Number', fontsize=18)  # y轴标签
    # 修改y轴刻度标签格式为"layer-数字"
    ax1.set_yticklabels([f'layer-{int(t.get_text())}' for t in ax1.get_yticklabels()])
    # 保持x轴刻度标签不变
    ax1.set_xticklabels([t.get_text() for t in ax1.get_xticklabels()])
    
    # 绘制vision模块热力图
    # 参数设置与text模块类似
    # cbar_ax=cbar_ax: 在指定位置显示颜色条
    sns.heatmap(vision_df, annot=True, fmt=".0f", cmap=custom_cmap, ax=ax2,
                linewidths=.5, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
    # 设置标题和轴标签
    ax2.set_title('Visual Encoder', fontsize=18, pad=12)
    ax2.set_xlabel('Projection Type', fontsize=18, labelpad=15)  # 增加labelpad值
    ax2.set_ylabel('Layer Number', fontsize=18)
    # 修改刻度标签格式
    ax2.set_yticklabels([f'layer-{int(t.get_text())}' for t in ax2.get_yticklabels()])
    ax2.set_xticklabels([t.get_text() for t in ax2.get_xticklabels()])
    
    # 调整坐标轴样式
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=0, labelsize=15)  # x轴标签旋转45度
        ax.tick_params(axis='y', rotation=0, labelsize=15)   # y轴标签水平显示
    
    # 设置整个图表的主标题
    # plt.suptitle('CLIP Rank Allocation Visualization', fontsize=24)
    # 自动调整子图之间的间距
    plt.tight_layout()
    
    # 构建输出文件路径并保存图像
    # dpi=300设置输出图像的分辨率
    output_path = f"output/awlora/figures/rank_visualization_{args.dataset}_{args.shots}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")

def visualize_rank_comparison(dataset, shots):
    """比较act和lod两种theta_type的rank分配情况
    Args:
        dataset: 数据集名称
        shots: few-shot数量
    """
    # 创建带颜色条布局的图形
    fig = plt.figure(figsize=(20, 16))
    # 创建2x3的网格布局
    # width_ratios=[0.95, 0.95, 0.05]: 两个图表和一个共享颜色条的宽度比
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 0.95, 0.1], height_ratios=[1, 1])
    
    # 创建子图
    ax1 = fig.add_subplot(gs[0, 0])  # act - Text encoder
    ax2 = fig.add_subplot(gs[1, 0])  # act - Vision encoder
    
    ax3 = fig.add_subplot(gs[0, 1])  # lod - Text encoder
    ax4 = fig.add_subplot(gs[1, 1])  # lod - Vision encoder
    cbar_ax = fig.add_subplot(gs[:, 2])  # 共享颜色条
    
    # 定义自定义颜色映射
    colors = ['#63C6F0', '#D2D7DA', '#F8C57D', '#E45D2D', '#394E86']
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

    # 存储所有数据的最大最小值
    all_vmins, all_vmaxs = [], []
    all_data = {}
    
    # 首先读取所有数据并确定全局颜色范围
    for theta_type in ['act', 'lod']:
        # 读取JSON文件
        json_path = f"output/awlora/json/rank_allocation_{theta_type}_{dataset}_{shots}.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 解析数据结构
        parsed_data = []
        for key, value in data.items():
            module_part, proj_part = key.split('.')
            parts = module_part.split('_')
            module_type = parts[0]
            block_num = parts[2]
            
            parsed_data.append({
                'Module': module_type.capitalize(),
                'Block': int(block_num),
                'Projection': proj_part,
                'Rank': int(value)
            })
        
        # 创建DataFrame
        df = pd.DataFrame(parsed_data)
        pivot_df = df.pivot_table(index=['Module', 'Block'], 
                                columns='Projection', 
                                values='Rank').sort_index(level=[0, 1])
        
        all_data[theta_type] = pivot_df
        
        # 分割数据
        text_df = pivot_df.xs('Text', level='Module')
        vision_df = pivot_df.xs('Vision', level='Module')
        
        # 收集最大最小值
        all_vmins.extend([text_df.min().min(), vision_df.min().min()])
        all_vmaxs.extend([text_df.max().max(), vision_df.max().max()])
    
    # 确定统一的颜色范围
    vmin, vmax = min(all_vmins), max(all_vmaxs)
    
    # 绘制所有子图
    for idx, (theta_type, ax_text, ax_vision) in enumerate([
        ('act', ax1, ax2),
        ('lod', ax3, ax4)
    ]):
        pivot_df = all_data[theta_type]
        text_df = pivot_df.xs('Text', level='Module')
        vision_df = pivot_df.xs('Vision', level='Module')
        
        # 绘制text模块热力图
        sns.heatmap(text_df, annot=True, fmt=".0f", cmap=custom_cmap, ax=ax_text,
                    linewidths=.5, cbar=False, vmin=vmin, vmax=vmax)
        ax_text.set_title(f'Text Encoder (θ={theta_type})', fontsize=18, pad=12)
        ax_text.set_xlabel('Projection Type', fontsize=18, labelpad=15)
        ax_text.set_ylabel('Layer Number', fontsize=18)
        
        # 绘制vision模块热力图
        sns.heatmap(vision_df, annot=True, fmt=".0f", cmap=custom_cmap, ax=ax_vision,
                    linewidths=.5, cbar=(True if idx == 1 else False), 
                    cbar_ax=(cbar_ax if idx == 1 else None),
                    vmin=vmin, vmax=vmax)
        ax_vision.set_title(f'Vision Encoder (θ={theta_type})', fontsize=18, pad=12)
        ax_vision.set_xlabel('Projection Type', fontsize=18, labelpad=15)
        ax_vision.set_ylabel('Layer Number', fontsize=18)
    
    # 设置颜色条字体大小
    cbar_ax.tick_params(labelsize=18)
    
    # 设置所有子图的刻度标签
    for ax in [ax1, ax2, ax3, ax4]:
        # 修改y轴刻度标签格式为"layer-数字"
        ax.set_yticklabels([f'layer-{int(t.get_text())}' for t in ax.get_yticklabels()])
        # 调整刻度标签大小和旋转角度
        ax.tick_params(axis='x', rotation=0, labelsize=15)
        ax.tick_params(axis='y', rotation=0, labelsize=15)
    
    # 设置整个图表的主标题
    plt.suptitle(f'CLIP Rank Allocation of {dataset} with {shots} shots', fontsize=24)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = f"output/awlora/figures/rank_visualization_{dataset}_{shots}.png"
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":    
    # json_path = f"output/awlora/json/rank_allocation_{args.theta_type}_{args.dataset}_{args.shots}.json"
    # visualize_rank_allocation(json_path)
    for dataset in ['dtd', 'stanford_cars', 'ucf101', 'food101', 'oxford_pets']:
        for shots in [4, 16]:
            visualize_rank_comparison(dataset, shots)
