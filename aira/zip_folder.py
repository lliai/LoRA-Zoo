import os
import zipfile

def zip_directory(path, output_filename):
    # 需要排除的文件夹
    exclude_dirs = {'data', 'draw', 'wandb'}
    
    # 创建zip文件
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历目录
        for root, dirs, files in os.walk(path):
            # 从遍历列表中移除需要排除的目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                file_path = os.path.join(root, file)
                # 获取相对路径，这样在zip文件中保持原有的目录结构
                arcname = os.path.relpath(file_path, path)
                zipf.write(file_path, arcname)

if __name__ == '__main__':
    # 当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # zip文件名
    zip_filename = 'project.zip'
    
    zip_directory(current_dir, zip_filename)
    print(f'已创建zip文件: {zip_filename}') 