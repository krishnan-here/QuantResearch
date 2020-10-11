
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
from datetime import datetime,timedelta
import ffn
import scipy
import math
import operator
import os


# In[ ]:


class summarystats:
    def __init__(self,region,datapath,outputpath):
        self.region=region
        self.datapath=datapath
        self.outputpath=outputpath
        

    def calcMMIndex(self,df,colname,idxname):
        df.loc[df.index[0],idxname]= 1
        prev_dt= df.index[0]
        for dt in df.index[1:]:
            caldays= (dt- prev_dt).days
            df.loc[dt,idxname]= df.loc[prev_dt,idxname]*(1+df.loc[prev_dt,colname]/360*caldays/100)
            prev_dt=dt
        df.drop(columns=colname,inplace=True)
        return df

    def getMMIndex(self):
        if (self.region=='US'):
            yld=web.DataReader('DGS1MO', 'fred',start='2000-01-01').dropna()## download 1-Month Treasury Constant Maturity Rate from FRB St louis
            yld.rename_axis(index={'DATE':'Date'},inplace=True)
            idx=self.calcMMIndex(yld.copy(),'DGS1MO','1MTBillIndex')
            
        if(self.region=='EUR'):
            yld= pd.read_csv(self.datapath+'\\1MEuribor.csv',skiprows=5,header=None).rename(columns={1:'Euribor'})
            yld['Date']= yld[0].apply(lambda x: pd.to_datetime(datetime.strptime(x,'%Y%b')))
            yld=yld.drop(columns=0).set_index('Date')
            idx= self.calcMMIndex(yld.copy(),'Euribor','1MEuriborIndex')
        return idx
    
    def rollingreturns(self,all_idxs,windows=[36,60]):
        mnth_end_rets= all_idxs.asfreq('M',method='ffill').pct_change()[1:]
        df= pd.DataFrame(columns=all_idxs.columns)
        rolling=  {}
        for window in windows:
            rolling[window]={}
            for k in ['Returns','Risk','Returns-Risk']:
                rolling[window][k]= pd.DataFrame(columns=all_idxs.columns)

            for i in range(window,len(mnth_end_rets)+1):
                idx= mnth_end_rets.index[i-1]
                rolling[window]['Returns'].loc[idx,:]=scipy.stats.gmean(1+mnth_end_rets.iloc[i-window:i,:])**12-1
                rolling[window]['Risk'].loc[idx,:]= mnth_end_rets.iloc[i-window:i,:].std()*np.sqrt(12)
                rolling[window]['Returns-Risk'].loc[idx,:]= rolling[window]['Returns'].loc[idx,:]/rolling[window]['Risk'].loc[idx,:]


            for k in ['Returns','Risk','Returns-Risk']:
                df.loc['Average '+str(window)+ 'months rolling returns',:]= np.round(100*rolling[window]['Returns'].mean(),2)
                df.loc['Average '+str(window)+ 'months rolling risk',:]= np.round(rolling[window]['Risk'].mean()*100,2)
                df.loc['Average '+str(window)+ 'months rolling return/risk',:]= np.round(rolling[window]['Returns-Risk'].mean().astype(float),2)

        return df,rolling


    def PerformanceSummaryWrapper(self,indexlevels,benchmark=True,simulationname=''):        

        indexnames=indexlevels.columns
        benchmarkname = indexnames[0]
        enddate=max(indexlevels.index)
        
        indexlevels= indexlevels.fillna(method='ffill').dropna()

        stats = ffn.core.GroupStats(indexlevels)
        Perf = stats.stats.loc[{'start','end','cagr','monthly_mean',
                                'monthly_vol','max_drawdown','monthly_skew','monthly_kurt','calmar'},
                                indexlevels.columns]
        RiskSummary = stats.stats.loc[{'start','end','monthly_vol','max_drawdown','monthly_skew','monthly_kurt','calmar'},
                                indexlevels.columns]
        RiskSummary.loc['start'] = [startdt.strftime('%Y-%m-%d') for startdt in RiskSummary.loc['start']]
        RiskSummary.loc['end'] = [enddt.strftime('%Y-%m-%d') for enddt in RiskSummary.loc['end']]

        drawdownseries = ffn.core.to_drawdown_series(indexlevels)

        RiskSummary.loc['Max Drawdown Period'] = [max(drawdownseries[(drawdownseries[column]==0)&
        (drawdownseries[column].index<min(drawdownseries[drawdownseries[column]==

        min(drawdownseries[column])].index))].index).strftime('%Y-%m-%d') + ' to '+
        max(drawdownseries[drawdownseries[column]==min(drawdownseries[column])].index).strftime('%Y-%m-%d')
        for column in indexlevels.columns]

        RiskSummary.loc['Max Downstreak Years (Absolute)'] = [max([x - drawdownseries[drawdownseries[column]==0].index[i - 1]
        for i, x in enumerate(drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
        )][1:]).days/365.0 for column in indexlevels.columns]

        RiskSummary.loc['Max Downstreak Period (Absolute)'] = [max(drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            [[np.argmax([x - drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            [i - 1] for i, x in enumerate(drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            )])-1]]).strftime('%Y-%m-%d')+' to '+
            max(drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            [[np.argmax([x - drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            [i - 1] for i, x in enumerate(drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            )])]]).strftime('%Y-%m-%d') for column in indexlevels.columns]

        rfr=pd.DataFrame()
        if (self.region=='US'):
            rfr = ffn.core.to_monthly(self.getMMIndex()).to_returns()[1:]
        elif(self.region=='EUR'):
            rfr= self.getMMIndex().to_returns()[1:]
        
        rfr.rename(columns={rfr.columns[0]:'Rtn'},inplace=True)
        rfr['Rtn'] = 1 + rfr['Rtn']

        # Calculate the geometric mean of risk-free rates from start-date to end-date
        Perf.loc['RFR'] = [scipy.stats.gmean(rfr['Rtn'][(rfr.index>start) & (rfr.index<=end)]) for (start,end) in zip(Perf.loc['start'], Perf.loc['end'])]
        Perf.loc['RFR'] = Perf.loc['RFR']**12 -1
        Perf.loc['Sharpe-Ratio'] = (Perf.loc['cagr'] - Perf.loc['RFR']) / Perf.loc['monthly_vol']


        Perf.loc['start'] = [startdt.strftime('%Y-%m-%d') for startdt in Perf.loc['start']]
        Perf.loc['end'] = [enddt.strftime('%Y-%m-%d') for enddt in Perf.loc['end']]
        Perf.loc['Return/Risk'] = Perf.loc['cagr'] / Perf.loc['monthly_vol']


     # round and multiply a few columns by 100
        Perf.loc[['cagr','monthly_mean','monthly_vol','max_drawdown'],:]= np.round(100*Perf.loc[['cagr','monthly_mean','monthly_vol','max_drawdown'],:].astype('float'),2)


        if benchmark:
            strategyreturns = ffn.core.to_monthly(indexlevels).to_returns()
            benchmarkreturns =  ffn.core.to_monthly(indexlevels[[benchmarkname]]).to_returns()
            excessreturns = strategyreturns - np.tile(benchmarkreturns,len(indexnames))
            gmreturns=strategyreturns+1

            relativeperformancelevels = (indexlevels.loc[:,indexlevels.columns[1:]] /np.transpose(np.tile(indexlevels.loc[:,benchmarkname],(len(indexnames)-1,1)))).rebase()

            drawdownseries =ffn.core.to_drawdown_series(relativeperformancelevels)

            RiskSummary.loc['Max Downstreak Years (Relative)'] = [0]+[max([x - drawdownseries[drawdownseries[column]==0].index[i - 1]
            for i, x in enumerate(drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            )][1:]).days/365.0 for column in indexlevels.columns[1:]]

            RiskSummary.loc['Max Downstreak Period (Relative)'] = ['']+[max(drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            [[np.argmax([x - drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            [i - 1] for i, x in enumerate(drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            )])-1]]).strftime('%Y-%m-%d')+' to '+
            max(drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            [[np.argmax([x - drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            [i - 1] for i, x in enumerate(drawdownseries[drawdownseries[column]==0].index.append(pd.DatetimeIndex([enddate]))
            )])]]).strftime('%Y-%m-%d') for column in indexlevels.columns[1:]]

            RiskSummary.loc['Downside Risk (%)']=np.round([math.sqrt(np.mean(np.square(np.minimum((strategyreturns[column] - np.mean(strategyreturns[column])),np.zeros(len((strategyreturns[column] -  np.mean(strategyreturns[column]))))))))*100*math.sqrt(12) for column in strategyreturns.columns],2)

            Perf.loc['Active Return (%)'] = Perf.loc['cagr'] - np.tile(Perf.loc['cagr',[benchmarkname]],len(indexnames))
            Perf.loc['Tracking Error (%)']= (excessreturns.std()*np.sqrt(12)*100).values
            Perf.loc['Tracking Error (%)',benchmarkname] = np.NaN

            Perf.loc['Information Ratio'] = Perf.loc['Active Return (%)'] /Perf.loc['Tracking Error (%)']
            RiskSummary.loc['Correlation'] = strategyreturns.corr()[benchmarkname]
            RiskSummary.loc['Beta'] = strategyreturns.cov()[benchmarkname] /np.tile(strategyreturns.var()[benchmarkname],len(indexnames))
            Perf.loc[['Active Return (%)','Tracking Error (%)','Information Ratio'],:]= np.round(Perf.loc[['Active Return (%)','Tracking Error (%)','Information Ratio'],:].astype('float'),2)
            RiskSummary.loc['Monthly Batting Average (%)']= np.round([x*100 for x in list(map(operator.truediv, [len(excessreturns[excessreturns[column]>0]) for column in excessreturns.columns], [len(excessreturns[column])-1 for column in excessreturns.columns]))],2)
            RiskSummary.loc['Upside Capture Ratio']= np.round([(scipy.stats.mstats.gmean(gmreturns[column] [gmreturns[benchmarkname]>1])-1)/(scipy.stats.mstats.gmean(gmreturns[benchmarkname] [gmreturns[benchmarkname]>1])-1) for column in gmreturns.columns],4)
            RiskSummary.loc['Downside Capture Ratio']= np.round([(scipy.stats.mstats.gmean(gmreturns[column][gmreturns[benchmarkname]<1])-1)/(scipy.stats.mstats.gmean(gmreturns[benchmarkname] [gmreturns[benchmarkname]<1])-1) for column in gmreturns.columns],4)

            RiskSummary.loc[['monthly_skew','monthly_kurt','Beta','calmar','Correlation','Max Downstreak Years (Absolute)',
            'Max Downstreak Years (Relative)'],:]= np.round(RiskSummary.loc[['monthly_skew','monthly_kurt','Beta',
            'Correlation','Max Downstreak Years (Absolute)','Max Downstreak Years (Relative)'],:].astype('float'),2)
            
            
            RiskSummary.loc[['max_drawdown','monthly_vol'],:]= np.round(100*RiskSummary.loc[['max_drawdown','monthly_vol'],:].astype('float'),2)

            RiskSummary = RiskSummary.loc[['start','end','monthly_vol','Downside Risk (%)','max_drawdown','calmar','Max Drawdown Period','Max Downstreak Years (Absolute)','Max Downstreak Period (Absolute)','Max Downstreak Years (Relative)',
                    'Max Downstreak Period (Relative)','Monthly Batting Average (%)','Upside Capture Ratio','Downside Capture Ratio','monthly_skew',\
                    'monthly_kurt','Correlation','Beta'],:]
            
            RiskSummary.rename(index={'max_drawdown':'Maximum Drawdown (%)',                          'monthly_vol':'Risk (%)','monthly_skew':'Skewness',                                     'monthly_kurt':'Kurtosis','calmar':'Calmar Ratio'},inplace=True)
            
        else:
            strategyreturns = ffn.core.to_monthly(indexlevels).to_returns()
            RiskSummary.loc['Downside Risk (%)']=np.round([math.sqrt(np.mean(np.square(np.minimum((strategyreturns[column] - np.mean(strategyreturns[column])),np.zeros(len((strategyreturns[column] -             np.mean(strategyreturns[column]))))))))*100*math.sqrt(12) for column in strategyreturns.columns],2)

            RiskSummary.loc[['monthly_skew','monthly_kurt','calmar'],:]= np.round(RiskSummary.loc[['monthly_skew','monthly_kurt','calmar'],:].astype('float'),2)

            RiskSummary.loc[['max_drawdown','monthly_vol'],:]= np.round(100*RiskSummary.loc[['max_drawdown','monthly_vol'],:].astype('float'),2)


            RiskSummary = RiskSummary.loc[['start','end','monthly_vol','Downside Risk (%)','max_drawdown',                                       'Max Drawdown Period','calmar','Max Downstreak Years (Absolute)',                                       'Max Downstreak Period (Absolute)','monthly_skew','monthly_kurt'],:]

            RiskSummary.rename(index={'max_drawdown':'Maximum Drawdown (%)',                      'monthly_vol':'Risk (%)','monthly_skew':'Skewness',                                 'monthly_kurt':'Kurtosis','calmar':'Calmar Ratio'},inplace=True)


        AdditionalPerf = Perf.loc[{'start','end'}]
        horizons = ['three_month','six_month','ytd','one_year','three_year','five_year','ten_year']
        commonhorizon = set(horizons) & set(stats.stats.index)
        commonhorizon = [ch for ch in horizons if ch in commonhorizon]
        horizonreturns = stats.stats.loc[commonhorizon,
                                indexlevels.columns]*100

        AdditionalPerf=AdditionalPerf.append(np.round(horizonreturns.astype('float'),2))
        calendaryearreturns = np.round(indexlevels.to_monthly().pct_change(periods=12)*100,2)
        calendaryearreturns = calendaryearreturns[calendaryearreturns.index.month==12].dropna()
        calendaryearreturns.index = calendaryearreturns.index.year
        AdditionalPerf = AdditionalPerf.append(calendaryearreturns)
        
        Perf.loc['Downside Risk (%)']=RiskSummary.loc['Downside Risk (%)']
        Perf.loc['Sortino-Ratio']= (Perf.loc['cagr'] - Perf.loc['RFR']) / Perf.loc['Downside Risk (%)']
        Perf.loc['Return/Max Drawdown']=Perf.loc['cagr']/np.abs(Perf.loc['max_drawdown'])
        Perf.loc[['Return/Risk','Sharpe-Ratio','Sortino-Ratio','monthly_skew','monthly_kurt','calmar','Return/Max Drawdown'],:]= np.round(Perf.loc[['Return/Risk','Sharpe-Ratio',                                                                                      'Sortino-Ratio','monthly_skew','monthly_kurt','calmar','Return/Max Drawdown'],:].astype('float'),2)

        Perf.loc[['Sortino-Ratio'],:]= np.round(Perf.loc[['Sortino-Ratio'],:].astype('float'),2)

        Perf = Perf.loc[['start','end','cagr','monthly_mean','monthly_vol','Downside Risk (%)','Return/Risk', 'monthly_skew',                         'monthly_kurt','Sharpe-Ratio','Sortino-Ratio',                         'max_drawdown','calmar','Return/Max Drawdown'],:]
        
        Perf.rename(index={'max_drawdown':'Maximum Drawdown (%)',                          'monthly_vol':'Risk (%)','cagr':'Annualized Compunded Return/CAGR(%)',                           'monthly_mean':'Annualized Arthimetic mean(%)','calmar':'Calmar Ratio',                          'monthly_skew':'Skewness',                                 'monthly_kurt':'Kurtosis'},inplace=True)


#         RiskSummary.index = [indexsummarylabels.get(indexname,indexname) for indexname in RiskSummary.index]
        
        simulname= self.region+'-Simulation-'+datetime.now().strftime('%Y%m%d-%H%M')+simulationname
#         os.mkdir(self.outputpath+'//results//'+simulname)
#         newpath=self.outputpath+'//results//'+simulname+'//'
        
        writer= pd.ExcelWriter(self.outputpath+simulname+'.xlsx')
        
        Perf.to_excel(writer,'PerformanceSummary')
#         Perf.to_csv(newpath+'PerformanceSummary.csv')
        RiskSummary.to_excel(writer,'RiskSummary')
#         RiskSummary.to_csv(newpath+'RiskSummary.csv')
        AdditionalPerf.to_excel(writer,'Horizon Returns')
    #         AdditionalPerf.to_csv(newpath+'Horizon Returns.csv')
        strategyreturns.to_excel(writer,'Strategy Returns')
#         strategyreturns.to_csv(self.outputpath+'strategyreturns.csv')
        strategyreturns.corr().to_excel(writer,'Correlation')
    
        dfroll,rolling= self.rollingreturns(indexlevels)
        dfroll.to_excel(writer,'Average Rolling Stats')
        for i in rolling.keys():
            for j in rolling[i].keys():
                rolling[i][j].to_excel(writer, 'rolling '+str(i)+'M '+str(j))
        
        
        writer.close()

#         strategyreturns.corr().to_csv(self.outputpath+'Correlation.csv')

        return Perf

