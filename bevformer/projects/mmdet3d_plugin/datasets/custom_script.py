import os.path
import pickle


def save_data_instance(instance, filename, index=0):
    """
    保存类对象到文件

    Parameters:
    - instance: 要保存的类对象实例
    - filename: 保存的文件名
    """
    name, ext = os.path.splitext(filename)
    save_name = name + '_idx_{:d}'.format(index) + ext
    with open(save_name, 'wb') as file:
        pickle.dump(instance, file)
    print(f"数据实例已保存到 {save_name}")


def load_data_instance(filename):
    """
    从文件加载类对象

    Parameters:
    - filename: 要加载的文件名

    Returns:
    - 类对象实例
    """
    with open(filename, 'rb') as file:
        instance = pickle.load(file)
    print(f"从 {filename} 加载数据实例成功")
    return instance
