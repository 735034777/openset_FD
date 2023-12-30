import os

def delete_file(file_path):
    """
    删除指定路径的文件

    Args:
        file_path (str): 要删除的文件的路径

    Returns:
        bool: 如果文件成功删除，则返回True；否则返回False。
    """
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"文件 {file_path} 已删除")
            return True
        except Exception as e:
            print(f"删除文件时发生错误: {e}")
            return False
    else:
        print(f"文件 {file_path} 不存在")
        return False