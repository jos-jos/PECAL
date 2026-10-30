# utils/logger.py
import csv
import os


class CSVLogger:
    def __init__(self, filepath, resume=False):
        """
        初始化 CSV Logger

        Args:
            filepath (str): 日志文件保存路径 (e.g., 'experiments/exp1/log.csv')
            resume (bool): 是否为断点续训模式。
                           - True: 以追加模式 ('a') 打开，不写表头
                           - False: 以写入模式 ('w') 打开，覆盖旧文件，写入表头
        """
        self.filepath = filepath
        self.headers = [
            'epoch',
            'train_loss', 'train_acc',
            'val_loss', 'val_acc',
            'val_f1', 'val_recall', 'val_precision',
            'time_sec',
            # 一致性约束消融（A/B/C/D）诊断字段
            'consistency_schedule', 'consistency_weight', 'same_class_rate',
        ]

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if not resume:
            # 新的一轮训练：覆盖旧文件，写入表头
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        else:
            # 断点续训：如果文件不存在，还是要写表头
            if not os.path.exists(self.filepath):
                with open(self.filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
            else:
                # ★ 表头兼容保护
                # 一致性消融新增了 3 个诊断字段。若直接向「改动前」产生的旧 log.csv
                # （9 列）追加 12 列的行，会造成列错位、文件损坏。
                # 这里检测到表头不一致时，把旧文件另存为 log_legacy.csv 并重建表头，
                # 旧数据不丢失，新行也保持列对齐。
                try:
                    with open(self.filepath, 'r', newline='', encoding='utf-8') as f:
                        first = f.readline().rstrip('\r\n')
                    existing = next(csv.reader([first]), []) if first else []
                    if existing and list(existing) != list(self.headers):
                        legacy = os.path.join(os.path.dirname(filepath),
                                              'log_legacy.csv')
                        n = 1
                        while os.path.exists(legacy):
                            legacy = os.path.join(os.path.dirname(filepath),
                                                  f'log_legacy_{n}.csv')
                            n += 1
                        os.replace(self.filepath, legacy)
                        print(f"   [logger] 检测到旧表头({len(existing)} 列) 与当前"
                              f"({len(self.headers)} 列) 不一致，"
                              f"旧日志已另存为 {os.path.basename(legacy)}")
                        with open(self.filepath, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(self.headers)
                except Exception as e:
                    print(f"   [logger] 表头检查失败（忽略）：{e}")

    def log(self, **kwargs):
        """
        记录一行日志

        Usage:
            logger.log(epoch=1, train_loss=0.5, val_acc=0.8, ...)
        """
        row = []
        for h in self.headers:
            # 如果传入了该字段的值，则使用；否则为空字符串或特定默认值
            val = kwargs.get(h, '')

            # 格式化浮点数，保留4位小数，看起来更整洁
            if isinstance(val, float):
                val = f"{val:.4f}"
            row.append(val)

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


# 简单的测试代码
if __name__ == "__main__":
    logger = CSVLogger("test_log.csv", resume=False)
    logger.log(epoch=1, train_loss=0.512345, val_acc=0.88, time_sec=12.5)
    logger.log(epoch=2, train_loss=0.412345, val_acc=0.92, time_sec=11.2)
    print("Test log saved to test_log.csv")