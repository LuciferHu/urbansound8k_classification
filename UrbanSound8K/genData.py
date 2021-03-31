# -*- coding:utf-8 -*-
"""
本文件用于生成训练集与验证集
各类别含有的示例数如下：
class: dog_bark has 960 examples.
class: children_playing has 978 examples.
class: car_horn has 408 examples.
class: air_conditioner has 1000 examples.
class: street_music has 932 examples.
class: gun_shot has 372 examples.
class: siren has 908 examples.
class: engine_idling has 1000 examples.
class: jackhammer has 999 examples.
class: drilling has 941 examples.
"""
import os
import struct
import pandas as pd


class GenUS8k(object):
    """
    读取CSV文件生成dataframe，为生成训练集与验证集提供数据来源。
    有两个csv文件来源，UrbanSound8K.csv是初始文件，需要对文件预筛。eight_labels_modifiy_classid.csv是已经经过筛选的数据总集，
    剔除了“car_horn”与“gun_shot”两个类别，并将“siren”与“street_music”的classid修改为“car_horn”与“gun_shot”的classid，读取该csv
    时不需要data_selection
    """
    def __init__(self, root):
        self.csv_root = root+'/metadata/eight_labels_modifiy_classid.csv'
        self.audio_root = root+'/audio'
        self.df_data = pd.read_csv(self.csv_root)  # 读取csv
        # self.data_selected = self._data_selection()  # 数据选择，当选择为eight_labels_modifiy_classid.csv时，不需要再筛选

    def _path_class(self, filename):
        """
        获取文件路径和标签值
        :param filename: 文件名
        :return: path_name, excerpt['class'].values[0]: 文件路径，类别，dtype: 字符串list
        """
        excerpt = self.df_data[self.df_data['slice_file_name'] == filename]
        # 检查文件名是否正确，如果错误，则寻找的是fold0，若正确，excerpt则获取到的是data['slice_file_name']，数据类型为dataFrame
        # print(excerpt)
        path_name = os.path.join(self.audio_root, 'fold' + str(excerpt.fold.values[0]), filename)
        # print(str(excerpt.fold.values[0]))

        return path_name, excerpt['class'].values[0]  # path_name是路径的字符串，单独对excerpt['class']转换成np.array，字符串也可以是numpy的数据内容

    def _wav_fmt_parser(self, filename):
        """
        获取音频文件的声道数、采样率、量化深度、并且将它们打包成一个tuple
        :return: (n_channels, s_rate, bit_depth)  tuple
        """
        full_path, _ = self._path_class(filename)
        wave_file = open(full_path, "rb")
        riff_fmt = wave_file.read(36)
        n_channels_string = riff_fmt[22:24]  # 声道数
        n_channels = struct.unpack("H", n_channels_string)[0]  # 声道数字节解包
        s_rate_string = riff_fmt[24:28]  # 采样率
        s_rate = struct.unpack("I", s_rate_string)[0]  # 采样率字节数据解包
        bit_depth_string = riff_fmt[-2:]  # 采样深度
        bit_depth = struct.unpack("H", bit_depth_string)[0]  # 采样深度字节数据解包

        return (n_channels, s_rate, bit_depth)  # 返回tuple: 声道数，采样率，采样深度

    def _add_attribution(self):
        """
        为data_frame添加属性，即声道数、采样率、采样深度
        :return: df_data
        """
        wav_fmt_data = [self._wav_fmt_parser(i) for i in self.df_data.slice_file_name]  # 是这8K数据的tuple array
        self.df_data[['n_channels', 'sampling_rate', 'bit_depth']] = pd.DataFrame(
            wav_fmt_data)  # 为data增加属性：声道数，采样率，采样深度
        return self.df_data

    def _data_selection(self):
        """
        挑选数据，要求采样率>=44100, 量化深度>=16bit
        :return: df_data
        """
        df_data_add_attr = self._add_attribution()
        df_data_selected = df_data_add_attr.loc[(df_data_add_attr["sampling_rate"] >= 44100) &    # 对采样率筛选
                                                (df_data_add_attr["bit_depth"] >= 16)] #&    # 对量化深度筛选
                                                # (df_data_add_attr['class'] != 'car_horn') &    # 剔除‘car_horn’类
                                                # (df_data_add_attr['class'] != 'gun_shot')]    # 剔除'gun_shot'类
        return df_data_selected


if __name__ == "__main__":
    df = GenUS8k(root="/home/richardhu/Documents/PycharmProjects/urbansound8k_classification/UrbanSound8K").df_data
    print(len(df['class'].unique()))
    print(df.shape)
    # shape=(7718,11)

