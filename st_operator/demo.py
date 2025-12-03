# Always leave yourself outs
# There's a lot more opportunities in the market than I was expecting.

import datetime
import akshare as ak
import time
from collections import deque
from security_data import *
import csv

id_dict={
    '600203':'frdz',
    '600776':'dftx',
    '600118':'zgwx',
    '002115':'swtx',
    '603686':'flm',
    '600580':'wldq',
    '603767':'zmcd',
    '000981':'szgk',
    '600537':'yjgd',
    '601162':'tfzq'
}
# id_list=['600203','600776','600118','002115','603686','600580','603767','000981','600537','601162']

# id_list=['601566','003401','000798','600756','000547','000407','002083','002264','600734']
# # jmw,zamj,zsyy,lcrj,htfz,slgf,frgf,xhd,sdjt
id_list=['600734','002264','000547','002792','002413']
#sdjt,xhd,htfz,tutx,lkfw
now = datetime.datetime.now()
today = now.date()
middle_time1 = datetime.datetime.combine(today, datetime.time(11, 30))
middle_time2 = datetime.datetime.combine(today, datetime.time(13, 00))
middle_time3 = datetime.datetime.combine(today, datetime.time(14, 30))
morning_time1 = datetime.datetime.combine(today, datetime.time(9, 25))
morning_time2 = datetime.datetime.combine(today, datetime.time(9, 30))

def gen_symbols(id_list):
    symbols=','.join([f'{id}.SZ' for id in id_list if id.startswith('00')])+','+','.join([f'{id}.SH' for id in id_list if id.startswith('60')])
    return symbols
now = datetime.datetime.now()
today = now.date()
filename = today.strftime("../results/history/hu%Y-%m-%d.cxv")
volume_dq=[deque(maxlen=60) for _ in range(len(id_list))]
volume_last=[0 for _ in range(len(id_list))]
volume_avg=[0 for _ in range(len(id_list))]
time_flag=False
attention_flag=False
attention_rate=0
volume_treshold=5000
with open(filename,'a',encoding='utf-8',newline='') as f:
    writer = csv.writer(f)

    for t in range(10000):    # ts.realtime_quote(ts_code='600000.SH,000001.SZ,000001.SH')
        symbols=gen_symbols(id_list)
        analyzer=ChipDistributionAnalyzer()
        df=analyzer.get_realtime_tick(ts_code=symbols)
        if df is None:
            print("can not get data")
        else:
            for i in range(len(df)):
                change_percent=(df.iloc[i,6]-df.iloc[i,5])/df.iloc[i,5]*100
                if change_percent<-7:
                    print(f"{df.iloc[i,1][1:7]}, {change_percent:.2f}, {df.iloc[i,23]}",end="\t")
                elif change_percent>8:
                    print(f"{df.iloc[i,1][1:7]}, {change_percent:.2f}, {df.iloc[i,13]}",end="\t")
                else:
                    print(f"{df.iloc[i,1][1:7]}, {change_percent:.2f}",end="\t")
                if t==0:
                    volume_last[i] = df.iloc[i, 11]
                elif t%10==0:
                    volume_now = df.iloc[i, 11]
                    volume_dq[i].append(volume_now-volume_last[i])
                    # 平均量为零，且队列中至少有10个数据点，则计算初始平均量
                    if volume_avg[i]==0:
                        if len(volume_dq[i])>=10:
                            volume_avg[i]=sum(volume_dq[i])/len(volume_dq[i])
                    # 否则，根据当前量和上一周期量更新平均量
                    else:
                        if volume_now-volume_last[i]>2.5*volume_avg[i]:
                            attention_rate = (volume_now - volume_last[i]) / volume_avg[i]
                            print(f"attention: {volume_now-volume_last[i]:.2f},{attention_rate:.2f}",end="\t")
                            attention_flag=True
                        # volume_avg[i]=(volume_avg[i]*59+(volume_now-volume_last[i]))/60
                        volume_avg[i]=sum(volume_dq[i])/len(volume_dq[i])
                        print(f"volume: {volume_now - volume_last[i]:.2f},{volume_avg[i]:.2f}", end="\t")
                        time_flag=True
                    volume_last[i] = volume_now

                print()
                buy_list=[df.iloc[i,13],df.iloc[i,15],df.iloc[i,17],df.iloc[i,19],df.iloc[i,21]]
                sel_list=[df.iloc[i,23],df.iloc[i,25],df.iloc[i,27],df.iloc[i,29],df.iloc[i,31]]
                if max(buy_list)>0.39*sum(buy_list) and max(buy_list)>10*min(buy_list) and max(buy_list)>volume_treshold:
                    if attention_flag:
                        writer.writerow([df.iloc[i,1],datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),df.iloc[i,6],*buy_list,*sel_list,f"attention {attention_rate:.2f}"])
                        attention_flag=False
                    else:
                        writer.writerow([df.iloc[i,1],datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),df.iloc[i,6],*buy_list,*sel_list])
                    print(f'record {buy_list,sel_list,t}')
                elif max(sel_list)>0.39*sum(sel_list) and max(sel_list)>10*min(sel_list) and max(sel_list)>volume_treshold:
                    if attention_flag:
                        writer.writerow([df.iloc[i, 1], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),df.iloc[i,6],*buy_list, *sel_list,f"attention {attention_rate:.2f}"])
                        attention_flag=False
                    else:
                        writer.writerow([df.iloc[i, 1], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),df.iloc[i,6],*buy_list, *sel_list])
                    print(f'record {buy_list, sel_list, t}')


        if t%20==0 or time_flag==True:
            print('time:',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            time_flag=False
        print('-'*15)
        time.sleep(3)
        now = datetime.datetime.now()
        if now>middle_time1 and now<middle_time2:
            print('middle time')
            time_diff=middle_time2-now
            sleep_time=min(time_diff.total_seconds(),30)
            time.sleep(time_diff.total_seconds())
        if now>middle_time3:
            volume_treshold=8000


# 需要看出承压能力


