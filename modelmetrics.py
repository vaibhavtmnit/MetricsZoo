from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import roc_curve, auc,precision_recall_curve,cohen_kappa_score,\
recall_score, precision_score, confusion_matrix
init_notebook_mode(connected=True) 


# This code is desgined to work on jupyter notebooks
class PerfMetrics():
    
    def __init__(self,y_true,y_score=None):
        
        #self.metric_type = metric_type
        self.y_true = y_true
        self.y_score = y_score
        


    def plots(self):
    
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_score[:,1])
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_score[:,1])

        auc_val = auc(tpr,fpr)
        fig = make_subplots(1,3,

                           subplot_titles = ('TPR/FPR vs. Threshold', f'AUC -{auc_val}','Precision-Recall Curve'))


        fig.add_scatter(x=thresholds,y=tpr, mode=None, name="TPR", row=1, col=1,
                       text='TPR/FPR vs. Threshold',textposition='top center')
        fig.add_scatter(x=thresholds,y=fpr, mode=None, name="FPR", row=1, col=1)
        fig.add_scatter(x=tpr,y=fpr, mode=None, name="AUC", row=1, col=2,fill='tonexty')
        fig.add_scatter(x=recall,y=precision ,mode=None, name="Precison-Recall", row=1, col=3,fill='tonexty')
        
        # Update xaxis properties
        fig.update_xaxes(title_text="Threshold", row=1, col=1)
        fig.update_xaxes(title_text="FPR", row=1, col=2)
        fig.update_xaxes(title_text="Recall", row=1, col=3)

        # Update yaxis properties
        fig.update_yaxes(title_text="TPR/FPR", row=1, col=1)
        fig.update_yaxes(title_text="TPR", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=2, col=1)


        fig.update_layout(title_text="Model Performance Plots",
                          title_font_size=30,
                         height=600,
                         width=1200)
        fig.show()
        
        
    def change_proba(self,y_score):
        
        self.y_score=y_score
        

    def conf_metric_at_k(self,threshold=0.5,label_values=None,get_vals=False):
        
        labels = self.get_labels(threshold=threshold)
        
        z = confusion_matrix(self.y_true,labels)

        
        if not(get_vals):
            x = labels_values if labels_values else [0,1]
            y = labels_values if labels_values else [0,1]

            fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='gray')
            fig.show()
            
        else:
            
            return z,labels

 
    def get_labels(self,threshold=0.5,maxval=False):
        
        y_score = self.y_score.values() if type(self.y_score) == pd.DataFrame else self.y_score
        
        if maxval:
            label = np.argmax(y_score,axis=1)          
        else:       
            #label = np.int16((y_score[:,0]<y_score[:,1]).reshape(-1,1)
            label = np.int16(y_score[:,1]>threshold).reshape(-1,1)
                             
        return label
                  
    
    def get_metrics(self,y_score=None,range_prob=[.5,.6,.7,.8,.9]):
        
        if y_score:
            self.y_score = y_score
        
        values=[]
        
        for prob in range_prob:
            
            cnf_mat,labels = self.conf_metric_at_k(prob,None, get_vals=True) 
            
            TN = cnf_mat[0][0]
            FN = cnf_mat[1][0]
            TP = cnf_mat[1][1]
            FP = cnf_mat[0][1]

            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP/(TP+FN)
            # Specificity or true negative rate
            TNR = TN/(TN+FP) 
            # Precision or positive predictive value
            PPV = TP/(TP+FP)
            # Negative predictive value
            NPV = TN/(TN+FN)
            # Fall out or false positive rate
            FPR = FP/(FP+TN)
            # False negative rate
            FNR = FN/(TP+FN)
            # False discovery rate
            FDR = FP/(TP+FP)
            # Overall accuracy
            ACC = ((TP+TN)/(TP+FP+FN+TN))*100
            
            Total = cnf_mat.sum()
            
            
            values.append([Total,FP,FN,TP,TN,TPR,FPR,ACC])
            
        df = pd.DataFrame(values,columns = ['Total','FP','FN','TP','TN','TPR','FPR','Accuracy %'])
        df.index=range_prob
        df.index.name = 'Threshold'

        s = df.style.format('{:.1f}')
        
            
                    
        cell_hover = {  # for row hover use <tr> instead of <td>
            'selector': 'td:hover',
            'props': [('background-color', '#ffffb3')]
        }
        
        index_names = {
        'selector': '.index_name',
        'props': [('background-color', '#000066'), ('color', 'white'), ('font-style','italic'), ('font-weight','normal')]
        }

        headers = {
            'selector': 'th:not(.index_name)',
            'props': [('background-color', '#000066'), ('color', 'white'), ('font-style','italic'), ('font-weight','normal')]
        }
        s.set_table_styles([cell_hover,headers,index_names])

        #return data
        return s
                       
            
    def score_distribution(self):
        
        data = pd.DataFrame({'y_true':self.y_true['song_popularity'],
              'y_pred':self.y_score[:,1]})
        
        data.sort_values(['y_pred'],ascending=False,inplace=True)
        
        data['Prob. Range']=pd.qcut(y_proba[:,1],q=10)
        
        data = data.groupby(['Prob. Range']).agg({'y_pred':len,
                                          'y_true':np.sum,
                                          })
        data.rename({'y_pred':'Total Obs.',
                    'y_true':'Total Response'},axis=1,inplace=True)
        data['No Response'] = data['Total Obs.']-data['Total Response']
        data['TP'] = data['Total Response'].cumsum()
        data['FP'] = data['Total Obs.'].cumsum() - data['TP']
        data['TN'] = (data['Total Obs.'].sum() - data['Total Response'].sum()) - data['FP']
        data['FN'] = data['Total Response'].sum() - data['TP']
        data['Precision'] = (data['TP'] / (data['TP'] +data['FP'] ))*100
        data['Recall'] = (data['TP']/data['Total Response'].sum())*100
        data['f1 score'] = (2*data['Precision']*data['Recall'])/(data['Precision']+data['Recall'])
        data['Accuracy'] = ((data['TP']+data['TN'])/data['Total Obs.'].sum())*100
        data['KS Stat.'] = (data['TP']/data['Total Response'].sum())*100 - \
        (data['FP']/(data['Total Response'].sum()-data['Total Response'].sum()))*100
         
        data.sort_index(ascending=False,inplace=True)
        
        s = data.style.format('{:.0f}')
                     
        cell_hover = {  # for row hover use <tr> instead of <td>
            'selector': 'td:hover',
            'props': [('background-color', '#ffffb3')]
        }
        
        index_names = {
        'selector': '.index_name',
        'props': [('background-color', '#000066'), ('color', 'white'), ('font-style','italic'), ('font-weight','normal')]
        }

        headers = {
            'selector': 'th:not(.index_name)',
            'props': [('background-color', '#000066'), ('color', 'white'), ('font-style','italic'), ('font-weight','normal')]
        }
        s.set_table_styles([cell_hover,headers,index_names])

        #return data
        return s
            
            
        
    # Use it to find if predictions from two models are similar or different
    def cohen_kappa(self,y1,y2,*args,**kwgs):
                
        return cohen_kappa_score(y1,y2,*args,**kwgs)
        
        
    