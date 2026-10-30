import os
import csv
from pathlib import Path
import argparse

def generate_csv_files(base_dir, output_dir=None):
    """
    为训练和验证数据集生成CSV文件
    
    参数:
    base_dir: 基础目录路径，包含trainval文件夹
    output_dir: 输出目录路径，默认为base_dir/trainval
    """
    # 设置基础路径
    base_path = Path(base_dir)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = base_path / "trainval"
    else:
        output_dir = Path(output_dir)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 预期的类别文件夹名称
    expected_labels = ["21", "22", "31", "41", "51", "61", "71", "72", "81", "91"]
    
    # 定义要处理的子集
    subsets = ["train", "val"]
    
    for subset in subsets:
        csv_data = []
        subset_path = base_path / "trainval" / subset
        
        if not subset_path.exists():
            print(f"警告: {subset_path} 目录不存在，跳过")
            continue
        
        print(f"处理 {subset} 数据集...")
        
        # 遍历每个类别文件夹
        for label in expected_labels:
            label_path = subset_path / label
            
            if not label_path.exists():
                print(f"警告: 类别文件夹 {label} 在 {subset} 中不存在")
                continue
            
            # 获取该类别下的所有图片文件
            # 支持的图片扩展名
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif'}
            
            for img_file in label_path.iterdir():
                if img_file.suffix.lower() in image_extensions and img_file.is_file():
                    # 使用相对于base_path的路径
                    rel_path = img_file.relative_to(base_path)
                    rel_path = os.path.join(base_dir, rel_path)
                    csv_data.append({
                        'image_path': str(rel_path),
                        'label': label
                    })
        
        # 按文件名排序，确保结果一致
        csv_data.sort(key=lambda x: x['image_path'])
        
        # 写入CSV文件
        csv_file = output_dir / f"{subset}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'label'])
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"已生成 {csv_file}，包含 {len(csv_data)} 条记录")
    
    return True

def verify_csv_files(csv_dir):
    """验证生成的CSV文件"""
    csv_dir = Path(csv_dir)
    
    for csv_file in ["train.csv", "val.csv"]:
        file_path = csv_dir / csv_file
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                print(f"{csv_file}: {len(rows)} 行数据")
                
                # 统计每个类别的数量
                label_counts = {}
                for row in rows:
                    label = row['label']
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                print(f"  类别分布: {label_counts}")
        else:
            print(f"文件不存在: {file_path}")

def main():
    parser = argparse.ArgumentParser(description='为wheat数据集生成训练和验证CSV文件')
    parser.add_argument('--base_dir', type=str, default='',
                       help='包含wheat文件夹的基目录，默认为当前目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出CSV文件的目录，默认为base_dir/trainval')
    parser.add_argument('--verify', action='store_true',
                       help='验证生成的CSV文件')
    
    args = parser.parse_args()
    
    # 设置wheat文件夹路径
    wheat_dir = Path(args.base_dir) / "wheat"
    
    if not wheat_dir.exists():
        print(f"错误: wheat文件夹不存在于 {wheat_dir}")
        print("请确保当前目录包含wheat文件夹，或使用--base_dir指定正确路径")
        return
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = wheat_dir / "trainval"
    else:
        output_dir = Path(args.output_dir)
    
    print(f"基础目录: {wheat_dir}")
    print(f"输出目录: {output_dir}")
    
    # 生成CSV文件
    try:
        success = generate_csv_files(wheat_dir, output_dir)
        if success:
            print("\nCSV文件生成完成!")
    except Exception as e:
        print(f"生成CSV文件时出错: {e}")
        return
    
    # 如果需要验证
    if args.verify:
        print("\n验证CSV文件:")
        verify_csv_files(output_dir)

# 测试函数
def test_with_sample_structure():
    """测试函数：创建示例目录结构并测试CSV生成"""
    import tempfile
    import shutil
    
    # 创建临时目录结构
    with tempfile.TemporaryDirectory() as tmpdir:
        print("创建测试目录结构...")
        tmpdir = Path(tmpdir)
        
        # 创建wheat目录结构
        wheat_dir = tmpdir / "wheat"
        trainval_dir = wheat_dir / "trainval"
        
        # 创建train和val目录
        train_dir = trainval_dir / "train"
        val_dir = trainval_dir / "val"
        
        # 创建类别文件夹
        labels = ["11", "21", "22", "31", "41", "51", "61", "71", "72", "81", "91", "99"]
        
        for label in labels:
            # 在train中创建类别文件夹和示例图片
            (train_dir / label).mkdir(parents=True, exist_ok=True)
            for i in range(3):  # 每个类别创建3个示例图片
                img_path = train_dir / label / f"train_{label}_{i}.jpg"
                img_path.write_text(f"模拟图片内容 {label}-{i}")
            
            # 在val中创建类别文件夹和示例图片
            (val_dir / label).mkdir(parents=True, exist_ok=True)
            for i in range(2):  # 每个类别创建2个示例图片
                img_path = val_dir / label / f"val_{label}_{i}.jpg"
                img_path.write_text(f"模拟图片内容 {label}-{i}")
        
        # 创建测试目录
        test_dir1 = wheat_dir / "O8562-21-22_test"
        test_dir2 = wheat_dir / "O3543-22-23_test"
        test_dir1.mkdir(exist_ok=True)
        test_dir2.mkdir(exist_ok=True)
        
        print(f"测试目录结构创建在: {wheat_dir}")
        
        # 生成CSV文件
        print("\n生成CSV文件...")
        success = generate_csv_files(wheat_dir, trainval_dir)
        
        if success:
            # 验证CSV文件
            print("\n验证生成的CSV文件:")
            verify_csv_files(trainval_dir)
            
            # 显示CSV文件内容示例
            for csv_file in ["train.csv", "val.csv"]:
                file_path = trainval_dir / csv_file
                if file_path.exists():
                    print(f"\n{csv_file} 前5行内容:")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:6]  # 包括标题行
                        for line in lines:
                            print(f"  {line.strip()}")

if __name__ == "__main__":
    # 如果直接运行，先进行测试
    print("开始生成CSV文件...")
    
    # 检查当前目录下是否有wheat文件夹
    if Path("wheat").exists():
        print("检测到当前目录下有wheat文件夹，开始处理...")
        main()
    else:
        print("当前目录下未找到wheat文件夹")
        print("1. 使用示例测试模式")
        print("2. 手动指定路径")
        
        choice = input("请选择 (1/2): ").strip()
        
        if choice == "1":
            test_with_sample_structure()
        else:
            base_dir = input("请输入包含wheat文件夹的基目录路径: ").strip()
            if not base_dir:
                base_dir = "."
            
            import sys
            sys.argv = [sys.argv[0], "--base_dir", base_dir, "--verify"]
            main()
