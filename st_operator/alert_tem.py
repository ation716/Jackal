from typing import Optional, Callable
import tkinter as tk
from tkinter import messagebox
import threading
import time


class SimpleDialog:
    """简化的弹窗工具类"""

    @staticmethod
    def auto_close(message: str,
                   title: str = "提示",
                   seconds: int = 5,
                   on_close: Optional[Callable] = None):
        """
        显示自动关闭弹窗

        示例:
            SimpleDialog.auto_close("保存成功！", seconds=3)
        """

        def show():
            root = tk.Tk()
            root.title(title)
            root.geometry("600x300")

            tk.Label(root, text=message, pady=30).pack()
            tk.Label(root, text=f"{seconds}秒后关闭").pack()

            def close():
                if on_close:
                    on_close()
                root.destroy()

            root.after(seconds * 1000, close)
            root.mainloop()

        threading.Thread(target=show, daemon=True).start()

    @staticmethod
    def confirm(message: str,
                title: str = "确认") -> bool:
        """
        显示确认弹窗

        示例:
            if SimpleDialog.confirm("确定要删除吗？"):
                delete_item()
        """
        root = tk.Tk()
        root.withdraw()
        result = messagebox.askyesno(title, message)
        root.destroy()
        return result


# 使用示例
if __name__ == "__main__":
    # 自动关闭弹窗（不阻塞主线程）
    SimpleDialog.auto_close("文件上传成功！", "成功", 3)
    SimpleDialog.auto_close("长时间弹窗测试", "成功", 30)
    time.sleep(40)
    # 确认弹窗（阻塞直到用户操作）
    # if SimpleDialog.confirm("确定要退出程序吗？"):
    #     print("程序退出")
    # else:
    #     print("继续运行")