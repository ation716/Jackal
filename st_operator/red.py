"""
remember every day
"""
import akshare as ak
import time
import datetime
import math
import os

def my_task():
    """你要周期执行的函数"""
    # 举例：返回当前时间字符串
    return ak.stock_market_activity_legu()

def run_schedule(interval_minutes=5):
    """从9:30到15:00，每interval_minutes分钟执行一次my_task"""
    now = datetime.datetime.now()
    today = now.date()

    # 定义时间区间
    start_time = datetime.datetime.combine(today, datetime.time(9, 30))
    end_time = datetime.datetime.combine(today, datetime.time(15, 0,8))
    middle_time1 = datetime.datetime.combine(today, datetime.time(11, 30,8))
    middle_time2 = datetime.datetime.combine(today, datetime.time(13, 00))

    # 如果当前时间在区间之外
    if now >= end_time:
        print("the end")
        return
    elif now < start_time:
        next_run = start_time
    else:
        # 当前时间超过9:30，找到最近的执行时间点（9:30 + n*x）
        minutes_since_start = (now - start_time).total_seconds() / 60
        n = math.ceil(minutes_since_start / interval_minutes)
        next_run = start_time + datetime.timedelta(minutes=n * interval_minutes)
        if next_run > end_time:
            print("the end")
            return

    # 文件名：当天日期，例如 2025-10-17.txt
    filename = today.strftime("../results/RED/em%Y-%m-%d.csv")

    print(f"let's start，Perform a task every {interval_minutes} minutes.")
    print(f"next task will start at {next_run.strftime('%H:%M:%S')}")

    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("")  # 创建空文件
        print(f"已创建文件: {filename}")

    while True:
        now = datetime.datetime.now()
        if now > end_time:
            print("到达15:00，停止执行。")
            break

        if now > next_run:

            df = my_task()

            # 如果文件存在 -> 追加模式（不写列名）
            first_column = df.iloc[:, [1]].T
            if datetime.datetime.strptime(first_column.iloc[0,11], '%Y-%m-%d %H:%M:%S')>=next_run:
                # 以追加模式写入，只有时间变化时才写入
                first_column.to_csv(filename, mode="a", index=False, header=False)
                next_run += datetime.timedelta(minutes=interval_minutes)
                print(now,next_run)
                print(first_column)

            # 如果在午休时间，跳过
            if next_run > middle_time1 and next_run<middle_time2:
                next_run=datetime.datetime.combine(today, datetime.time(13, 00))
                continue


            if next_run > end_time:
                print("the end")
                break

        time_diff = next_run-now
        seconds = max(time_diff.total_seconds(),10)
        time.sleep(seconds)

if __name__ == "__main__":
    run_schedule(interval_minutes=5)

