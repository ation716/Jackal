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

id_list=['002171','603686','002213','002512']
# cjxc,flm,dwgf,dhzn

def gen_symbols(id_list):
    symbols=','.join([f'{id}.SZ' for id in id_list if id.startswith('00')])+','+','.join([f'{id}.SH' for id in id_list if id.startswith('60')])
    return symbols
now = datetime.datetime.now()
today = now.date()
filename = today.strftime("../results/history/hu%Y-%m-%d.cxv")
volume_dq=[deque(maxlen=60) for _ in range(len(id_list))]
volume_last=[0 for _ in range(len(id_list))]
volume_avg=[0 for _ in range(len(id_list))]
with open(filename,'a',encoding='utf-8') as f:
    writer = csv.writer(f)
    pass

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
                volume_dq[i].append(volume_last[i]-volume_now)
                # 平均量为零，且队列中至少有10个数据点，则计算初始平均量
                if volume_avg[i]==0:
                    if len(volume_dq[i])>=10:
                        volume_avg[i]=sum(volume_dq[i])/len(volume_dq[i])
                # 否则，根据当前量和上一周期量更新平均量
                else:
                    if volume_last[i]-volume_now>2.5*volume_avg[i]:
                        print(f"attention: {volume_last[i]-volume_now:.2f},{(volume_now-volume_last[i])/volume_avg[i]:.2f}",end="\t")
                    volume_avg[i]=(volume_avg[i]*59+(volume_last[i]-volume_now))/60

                volume_last[i] = volume_now

            print()
            buy_list=[df.iloc[i,13],df.iloc[i,15],df.iloc[i,17],df.iloc[i,19],df.iloc[i,21]]
            sel_list=[df.iloc[i,23],df.iloc[i,25],df.iloc[i,27],df.iloc[i,29],df.iloc[i,31]]
            if max(buy_list)>0.35*sum(buy_list) and max(buy_list)>5*min(buy_list):
                writer.writerow([df.iloc[i,1],datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),*buy_list,*sel_list])
                print(f'record {buy_list,sel_list,t}')
            elif max(sel_list)>0.35*sum(sel_list) and max(sel_list)>5*min(sel_list):
                writer.writerow([df.iloc[i, 1], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *buy_list, *sel_list])
                print(f'record {buy_list, sel_list, t}')


    if t%20==0:
        print('time:',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('-'*15)
    time.sleep(3)

# 需要看出承压能力


